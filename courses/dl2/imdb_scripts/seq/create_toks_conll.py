from fastai.text import *
import fire
from create_toks import fixup

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

BOS_LABEL = '_bos_'
PAD = '_pad_'

re1 = re.compile(r'  +')


def read_file(filepath):
    assert os.path.exists(filepath)
    sentences = []
    labels = []
    with open(filepath, encoding='utf-8') as f:
        sentence = [BOS]
        sentence_labels = [BOS_LABEL]
        for line in f:
            if line == '\n':
                sentences.append(sentence)
                labels.append(sentence_labels)
                sentence = [BOS]  # use xbos as the start of sentence token
                sentence_labels = [BOS_LABEL]
            else:
                sentence.append(fixup(line.split()[0].lower()))
                # label is generally in the last column
                sentence_labels.append(line.split()[-1])
        if sentence:  # some files, e.g. NER end on an empty line
            sentences.append(sentence)
            labels.append(sentence_labels)
    return sentences, labels


def create_toks(prefix, max_vocab=30000, min_freq=1):
    PATH = f'data/nlp_seq/{prefix}/'

    names = {}
    if prefix == 'ner':
        names['train'] = 'train.txt'
        names['val'] = 'valid.txt'
        names['test'] = 'test.txt'
    else:
        raise ValueError(f'Filenames for {prefix} have to be added first.')
    paths = {}
    for split in ['train', 'val', 'test']:
        paths[split] = f'{PATH}{names[split]}'

    print(f'prefix {prefix} max_vocab {max_vocab} min_freq {min_freq}')

    os.makedirs(f'{PATH}tmp', exist_ok=True)
    trn_tok, trn_labels = read_file(paths['train'])
    val_tok, val_labels = read_file(paths['val'])
    test_tok, test_labels = read_file(paths['test'])

    for trn_t, trn_l in zip(trn_tok[:5], trn_labels[:5]):
        print('Sentence:', trn_t, 'labels:', trn_l)

    print(f'# of train: {len(trn_tok)}, # of val: {len(val_tok)},'
          f'# of test: {len(test_tok)}')

    freq = Counter(p for o in trn_tok for p in o)
    print(freq.most_common(25))
    itos = [o for o, c in freq.most_common(max_vocab) if c > min_freq]
    itos.insert(0, PAD)
    itos.insert(0, '_unk_')
    stoi = collections.defaultdict(lambda: 0,
                                   {v: k for k, v in enumerate(itos)})
    print(len(itos))

    trn_ids = np.array([[stoi[o] for o in p] for p in trn_tok])
    val_ids = np.array([[stoi[o] for o in p] for p in val_tok])
    test_ids = np.array([[stoi[o] for o in p] for p in test_tok])

    # map the labels to ids
    freq = Counter(p for o in trn_labels for p in o)
    print(freq)
    itol = [l for l, c in freq.most_common()]
    itol.insert(1, PAD)  # insert padding label at index 1
    print(itol)
    ltoi = {l: i for i, l in enumerate(itol)}
    trn_lbl_ids = np.array([[ltoi[o] for o in p] for p in trn_labels])
    val_lbl_ids = np.array([[ltoi[o] for o in p] for p in val_labels])
    test_lbl_ids = np.array([[ltoi[o] for o in p] for p in test_labels])

    ids_joined = np.array([[stoi[o] for o in p] for p in trn_tok + val_tok + test_tok])
    val_ids_joined = ids_joined[int(len(ids_joined)*0.9):]
    ids_joined = ids_joined[:int(len(ids_joined)*0.9)]

    np.save(f'{PATH}tmp/trn_ids.npy', trn_ids)
    np.save(f'{PATH}tmp/val_ids.npy', val_ids)
    np.save(f'{PATH}tmp/test_ids.npy', test_ids)
    np.save(f'{PATH}tmp/lbl_trn.npy', trn_lbl_ids)
    np.save(f'{PATH}tmp/lbl_val.npy', val_lbl_ids)
    np.save(f'{PATH}tmp/lbl_test.npy', test_lbl_ids)
    pickle.dump(itos, open(f'{PATH}tmp/itos.pkl', 'wb'))
    pickle.dump(itol, open(f'{PATH}tmp/itol.pkl', 'wb'))
    np.save(f'{PATH}tmp/trn_lm_ids.npy', ids_joined)
    np.save(f'{PATH}tmp/val_lm_ids.npy', val_ids_joined)


if __name__ == '__main__': fire.Fire(create_toks)
