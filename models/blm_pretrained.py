from .blm import BLM
from transformers import AutoModel, AutoModelForMaskedLM
import pytorch_lightning as pl
import torch.nn as nn
from . torch_utils import get_canvas, sample_permutation, seq_cross_entropy, collect, batch_randint, select
import torch.nn.functional as F
import torch


class PBLM(BLM):
    def __init__(self, hparams):
        super(PBLM, self).__init__(hparams)
        self.hparams=hparams

        self.enc=AutoModel.from_pretrained(hparams.plm_name)

        if hparams.share_emb_prj_weight:
            self.word.weight = self.enc.embeddings.word_embeddings.weight
            self.x_logit_scale = (hparams.d_model ** -0.5)
    
    def forward_encoder(self, canvas):
        output=self.enc(canvas)
        return output.last_hidden_state
    
    def get_loss(self, seq, canvas, blanks, rest, loc, lb, rb):
        count = (rest != -1).sum(1)
        output = self.forward_encoder(canvas)
        output_blank = collect(output, blanks)

        logits_loc = self.loc(output_blank).squeeze(-1)
        logits_loc[blanks == -1] = float('-inf')
        nll_loc = -F.log_softmax(logits_loc, 1)
        loss_loc = collect(nll_loc, loc)
        loss_loc = loss_loc.sum(1) / count.float()
        output_loc = collect(output_blank, loc)

        logits_word = self.word(output_loc) * self.x_logit_scale
        target = collect(seq, rest, 0)
        loss_word = seq_cross_entropy(logits_word, target, 0)
        loss_word = loss_word.sum(1) / count.float()
        output_word = torch.cat((output_loc, self.enc.embeddings.word_embeddings(target)), -1)

        logits_lrb = self.lrb(output_word)
        loss_lrb = seq_cross_entropy(logits_lrb, lb * 2 + rb, -3)
        loss_lrb = loss_lrb.sum(1) / count.float()

        return loss_loc, loss_word, loss_lrb