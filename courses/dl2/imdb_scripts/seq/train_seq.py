import fire
from fastai.text import *
from fastai.lm_rnn import *

from eval import eval_ner


def freeze_all_but(learner, n):
    c=learner.get_layer_groups()
    for l in c: set_trainable(l, False)
    set_trainable(c[n], True)


def get_rnn_seq_labeler(bptt, max_seq, n_class, n_tok, emb_sz, n_hid, n_layers, pad_token, layers, drops, bidir=False,
                      dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5):
    rnn_enc = MultiBatchSeqRNN(bptt, max_seq, n_tok, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
                      dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop)
    # return SequentialRNN(rnn_enc, LinearBlocks(layers, drops))
    return SequentialRNN(rnn_enc, LinearDecoder(n_class, emb_sz, 0.1))


class MultiBatchSeqRNN(RNN_Encoder):
    def __init__(self, bptt, max_seq, *args, **kwargs):
        self.max_seq,self.bptt = max_seq,bptt
        super().__init__(*args, **kwargs)

    def concat(self, arrs):
        return [torch.cat([l[si] for l in arrs]) for si in range(len(arrs[0]))]

    def forward(self, input):
        sl,bs = input.size()
        for l in self.hidden:
            for h in l: h.data.zero_()
        # raw_outputs, outputs = [],[]
        raw_outputs, outputs = super().forward(input)
        # for i in range(0, sl, self.bptt):
        #     r, o = super().forward(input[i: min(i+self.bptt, sl)])
        #     if i>(sl-self.max_seq):
        #         raw_outputs.append(r)
        #         outputs.append(o)
        # return self.concat(raw_outputs), self.concat(outputs)
        return raw_outputs, outputs


class SeqDataLoader(DataLoader):
    def get_batch(self, indices):
        res = self.np_collate([self.dataset[i] for i in indices])
        # res = self.np_collate([self.dataset[i] for i in indices], self.pad_idx)
        # if not self.transpose: return res
        # res[0] = res[0].T
        # print('First seq:', res[0][0])
        # print('First labels:', res[1][0])
        res[1] = np.reshape(res[1], -1)  # reshape the labels to one sequence
        return res


class TextSeqDataset(Dataset):
    def __init__(self, x, y, backwards=False, sos=None, eos=None):
        self.x,self.y,self.backwards,self.sos,self.eos = x,y,backwards,sos,eos

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]  # we need to get y as array
        if self.backwards: x = list(reversed(x))
        if self.eos is not None: x = x + [self.eos]
        if self.sos is not None: x = [self.sos]+x
        return np.array(x),np.array(y)

    def __len__(self): return len(self.x)


def train_seq(dir_path, cuda_id, lm_id='', clas_id=None, bs=64, cl=1, backwards=False, startat=0, unfreeze=True,
              lr=0.01, dropmult=1.0, pretrain=True, bpe=False, use_clr=True,
              use_regular_schedule=False, use_discriminative=True, last=False, chain_thaw=False,
              from_scratch=False, train_file_id=''):
    print(f'prefix {dir_path}; cuda_id {cuda_id}; lm_id {lm_id}; clas_id {clas_id}; bs {bs}; cl {cl}; backwards {backwards}; '
        f'dropmult {dropmult} unfreeze {unfreeze} startat {startat}; pretrain {pretrain}; bpe {bpe}; use_clr {use_clr};'
        f'use_regular_schedule {use_regular_schedule}; use_discriminative {use_discriminative}; last {last};'
        f'chain_thaw {chain_thaw}; from_scratch {from_scratch}; train_file_id {train_file_id}')

    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)
    PRE = 'bwd_' if backwards else 'fwd_'
    PRE = 'bpe_' + PRE if bpe else PRE
    IDS = 'bpe' if bpe else 'ids'
    dir_path = Path(dir_path)
    train_file_id = train_file_id if train_file_id == '' else f'_{train_file_id}'
    lm_id = lm_id if lm_id == '' else f'{lm_id}_'
    clas_id = lm_id if clas_id is None else clas_id
    clas_id = clas_id if clas_id == '' else f'{clas_id}_'
    lm_file = f'{PRE}{lm_id}lm_enc'
    lm_path = dir_path / 'models' / f'{lm_file}.h5'
    if not from_scratch:
        assert lm_path.exists(), f'Error: {lm_path} does not exist.'
    # bptt,em_sz,nh,nl = 70,400,1150,3
    bptt, em_sz, nh, nl = 70, 100, 100, 2

    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

    if backwards:
        trn_sent = np.load(dir_path / 'tmp' / f'trn_{IDS}{train_file_id}_bwd.npy')
        val_sent = np.load(dir_path / 'tmp' / f'val_{IDS}_bwd.npy')
        test_sent = np.load(dir_path / 'tmp' / f'test_{IDS}_bwd.npy')
    else:
        trn_sent = np.load(dir_path / 'tmp' / f'trn_{IDS}{train_file_id}.npy')
        val_sent = np.load(dir_path / 'tmp' / f'val_{IDS}.npy')
        test_sent = np.load(dir_path / 'tmp' / f'test_{IDS}.npy')

    trn_lbls = np.load(dir_path / 'tmp' / f'lbl_trn{train_file_id}.npy')
    val_lbls = np.load(dir_path / 'tmp' / f'lbl_val.npy')
    test_lbls = np.load(dir_path / 'tmp' / f'lbl_test.npy')
    id2label = pickle.load(open(dir_path / 'tmp' / 'itol.pkl', 'rb'))
    c = len(id2label)

    if bpe:
        vs=30002
    else:
        id2token = pickle.load(open(dir_path / 'tmp' / 'itos.pkl', 'rb'))
        vs = len(id2token)

    print('Train sentences shape:', trn_sent.shape)
    print('Train labels shape:', trn_lbls.shape)
    print('Token ids:', [id2token[id_] for id_ in trn_sent[0]])
    print('Label ids:', [id2label[id_] for id_ in trn_lbls[0]])

    trn_ds = TextSeqDataset(trn_sent, trn_lbls)
    val_ds = TextSeqDataset(val_sent, val_lbls)
    test_ds = TextSeqDataset(test_sent, test_lbls)
    trn_samp = SortishSampler(trn_sent, key=lambda x: len(trn_sent[x]), bs=bs//2)
    val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
    test_samp = SortSampler(test_sent, key=lambda x: len(test_sent[x]))
    trn_dl = SeqDataLoader(trn_ds, bs//2, transpose=False, num_workers=1, pad_idx=1, sampler=trn_samp)  # TODO why transpose? Should we also transpose the labels?
    val_dl = SeqDataLoader(val_ds, bs, transpose=False, num_workers=1, pad_idx=1, sampler=val_samp)
    test_dl = SeqDataLoader(test_ds, bs, transpose=False, num_workers=1, pad_idx=1, sampler=test_samp)
    md = ModelData(dir_path, trn_dl, val_dl, test_dl)

    dps = np.array([0.4,0.5,0.05,0.3,0.4])*dropmult
    #dps = np.array([0.5, 0.4, 0.04, 0.3, 0.6])*dropmult
    #dps = np.array([0.65,0.48,0.039,0.335,0.34])*dropmult
    #dps = np.array([0.6,0.5,0.04,0.3,0.4])*dropmult

    m = get_rnn_seq_labeler(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
              layers=[em_sz, 50, c], drops=[dps[4], 0.1],
              dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])

    learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
    learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learn.clip=25.
    learn.metrics = [accuracy]

    lrm = 2.6
    if use_discriminative:
        lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])
    else:
        lrs = lr
    wd = 1e-6
    if not from_scratch:
        print(f'Loading encoder from {lm_file}...')
        learn.load_encoder(lm_file)
    else:
        print('Training classifier from scratch. LM encoder is not loaded.')
        use_regular_schedule = True

    if (startat<1) and pretrain and not last and not chain_thaw and not from_scratch:
        learn.freeze_to(-1)
        learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 1,
                  use_clr=None if use_regular_schedule or not use_clr else (8,3))
        learn.freeze_to(-2)
        learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 1,
                  use_clr=None if use_regular_schedule or not use_clr else (8, 3))
        learn.save(f'{PRE}{clas_id}clas_0')
    elif startat==1:
        learn.load(f'{PRE}{clas_id}clas_0')

    if chain_thaw:
        lrs = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.001])
        print('Using chain-thaw. Unfreezing all layers one at a time...')
        n_layers = len(learn.get_layer_groups())
        print('# of layers:', n_layers)
        # fine-tune last layer
        learn.freeze_to(-1)
        print('Fine-tuning last layer...')
        learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 1,
                  use_clr=None if use_regular_schedule or not use_clr else (8,3))
        n = 0
        # fine-tune all layers up to the second-last one
        while n < n_layers-1:
            print('Fine-tuning layer #%d.' % n)
            freeze_all_but(learn, n)
            learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 1,
                      use_clr=None if use_regular_schedule or not use_clr else (8,3))
            n += 1

    if unfreeze:
        learn.unfreeze()
    else:
        learn.freeze_to(-3)

    if last:
        print('Fine-tuning only the last layer...')
        learn.freeze_to(-1)

    if use_regular_schedule:
        print('Using regular schedule. Setting use_clr=None, n_cycles=cl, cycle_len=None.')
        use_clr = None
        n_cycles = cl
        cl = None
    else:
        n_cycles = 1
    learn.fit(lrs, n_cycles, wds=wd, cycle_len=cl, use_clr=(8,8) if use_clr else None)
    print('Plotting lrs...')
    learn.sched.plot_lr()
    learn.save(f'{PRE}{clas_id}clas_1')

    eval_ner(learn, id2label, is_test=False)
    eval_ner(learn, id2label, is_test=True)

if __name__ == '__main__': fire.Fire(train_seq)
