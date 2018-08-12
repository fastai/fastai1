import fire
from fastai.text import *
from fastai.lm_rnn import *

from train_seq import get_rnn_seq_labeler, TextSeqDataset, SeqDataLoader
from eval import eval_ner


def evaluate(dir_path, cuda_id, clas_id='', bs=64, bpe=False):

    print(f'prefix {dir_path}; cuda_id {cuda_id}; bs {bs}; bpe {bpe}')
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)
    PRE_FWD = 'fwd_'
    PRE_FWD = 'bpe_' + PRE_FWD if bpe else PRE_FWD
    IDS = 'bpe' if bpe else 'ids'
    if clas_id != '': clas_id += '_'
    dir_path = Path(dir_path)
    fwd_clas_file = f'{PRE_FWD}{clas_id}clas_1'
    fwd_clas_path = dir_path / 'models' / f'{fwd_clas_file}.h5'
    assert fwd_clas_path.exists(), f'Error: {fwd_clas_path} does not exist.'

    bptt,em_sz,nh,nl = 70,400,1150,3
    opt_fn = partial(optim.Adam, betas=(0.7, 0.99))

    trn_sent = np.load(dir_path / 'tmp' / f'trn_{IDS}.npy')
    val_sent = np.load(dir_path / 'tmp' / f'val_{IDS}.npy')
    test_sent = np.load(dir_path / 'tmp' / f'test_{IDS}.npy')

    trn_lbls = np.load(dir_path / 'tmp' / 'lbl_trn.npy')
    val_lbls = np.load(dir_path / 'tmp' / 'lbl_val.npy')
    test_lbls = np.load(dir_path / 'tmp' / f'lbl_test.npy')
    id2label = pickle.load(open(dir_path / 'tmp' / 'itol.pkl', 'rb'))
    print('id2label:', id2label)
    c = len(id2label)

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

    if bpe: vs=30002
    else:
        itos = pickle.load(open(dir_path / 'tmp' / f'itos.pkl', 'rb'))
        vs = len(itos)

    dps = np.array([0.4,0.5,0.05,0.3,0.4])*0.9

    m = get_rnn_seq_labeler(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
              layers=[em_sz, 50, c], drops=[dps[4], 0.1],
              dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])

    learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
    learn.load(fwd_clas_file)

    eval_ner(learn, id2label, is_test=False)
    eval_ner(learn, id2label, is_test=True)

if __name__ == '__main__': fire.Fire(evaluate)
