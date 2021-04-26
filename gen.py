from utils import load_model
from vocab import Vocab
import os

checkpoint_file="checkpoints/yelp/neg/lightning_logs/version_0/checkpoints/model.ckpt"

model = load_model(checkpoint_file).to('cpu')
model.eval()
vocab = Vocab(os.path.join(model.hparams.root_dir, 'vocab.txt'))

input="___ place ___ and ___ food ___ .".lower()
s=input.replace("___", "<blank>").split()
s += ['<eos>'] if model.hparams.add_eos else []
s = [vocab.word_to_idx(w) for w in s]
_, full = model.generate(s, 'greedy', 'cpu')
full = [[vocab.idx2word[id] for id in ids] for ids in full]
for step in full:
    print(" ".join(step).replace("<blank>", "\_\_\_"))