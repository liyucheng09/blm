import os
import yaml
from vocab import Vocab
from tqdm import tqdm
from models import get_model_class
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def strip_eos(sents):
    return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
            for sent in sents]

def makedir(path):
    dir = os.path.dirname(path)
    if dir:
        os.makedirs(dir, exist_ok=True)


def repeat(f, x, n):
    for i in range(n):
        x = f(x)
    return x


def get_hparams(checkpoint):
    hparams_file = os.path.join(os.path.dirname(os.path.dirname(checkpoint)), 'hparams.yaml')
    with open(hparams_file) as stream:
        return yaml.safe_load(stream)


def load_model(checkpoint):
    hparams = get_hparams(checkpoint)
    model = get_model_class(hparams['model_type']).load_from_checkpoint(checkpoint, hparams=hparams)
    model.hparams.root_dir = repeat(lambda x: os.path.dirname(x), checkpoint, 4)
    return model


def load_sent(path, add_eos=False):
    sents = []
    with open(path) as f:
        for line in f:
            s = line.split()
            if add_eos:
                s += ['<eos>']
            sents.append(s)
    return sents

def load_sent_with_tokenizer(path, add_eos=False):
    sents=[]
    with open(path) as f:
        for line in tqdm(f):
            ids = tokenizer(line, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False)['input_ids']
            s = [tokenizer.convert_ids_to_tokens(i) for i in ids ]
            if add_eos:
                s += ['<eos>']
            sents.append(s)
    return sents

def load_data(path, add_eos=False, cat_sent=False, max_len=512, tokenizer_type=None):
    if tokenizer_type is not None:
        global tokenizer
        from transformers import AutoTokenizer, BertTokenizer
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_type)

    if not add_eos:
        print('WARNING: You should always use add_eos to get comparable PPL on '
              'language modeling tasks')

    sents = load_sent(path, add_eos) if not tokenizer else load_sent_with_tokenizer(path, add_eos)
    if cat_sent:
        if not add_eos:
            raise ValueError('Using cat_sent without add_eos')
        d = [w for s in sents for w in s]
        data = [d[i: i + max_len] for i in range(0, len(d), max_len)]
    else:
        print('# truncated sentences:',
              sum(1 for s in sents if len(s) > max_len))
        data = [s[:max_len] for s in sents]
    return data


def write(file, sents, write_mid):
    sents = strip_eos(sents)
    if write_mid:
        for s in sents:
            file.write(' '.join(s) + '\n')
        file.write('\n')
    else:
        file.write(' '.join(sents[-1]) + '\n')
    file.flush()

def gen(model, vocab):
    input="___ place ___ and ___ food ___ .".lower()
    s=input.replace("___", "<blank>").split()
    s += ['<eos>'] if model.hparams.add_eos else []
    s = [vocab.word_to_idx(w) for w in s]
    _, full = model.generate(s, 'greedy', 'cpu')
    full = [[vocab.idx2word[id] for id in ids] for ids in full]
    for step in full:
        print(" ".join(step).replace("<blank>", "\_\_\_"))

def get_model(checkpoint_file, device):
    model = load_model(checkpoint_file).to(device)
    model.eval()
    vocab = Vocab(os.path.join(model.hparams.root_dir, 'vocab.txt'))
    return model, vocab

def get_early_stopping_callback():
    return EarlyStopping(
        monitor='train_loss',
        patience=40,
        mode='min',
    )