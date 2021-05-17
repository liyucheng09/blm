from .blm import BLM
from transformers import AutoModel, AutoModelForMaskedLM
import pytorch_lightning as pl
import torch.nn as nn
from . torch_utils import get_canvas, sample_permutation, seq_cross_entropy, collect, batch_randint, select
import torch.nn.functional as F
import torch
from lyc.utils import get_optimizer_and_schedule
from .lm import LM


class PBLM(BLM):
    def __init__(self, hparams):
        super(LM, self).__init__()
        self.hparams=vars(hparams)

        self.word = nn.Linear(hparams.d_model, hparams.vocab_size, bias=False)
        nn.init.xavier_normal_(self.word.weight)
        self.x_logit_scale = 1.

        self.enc=AutoModel.from_pretrained(hparams.model_name_or_path, cache_dir=hparams.cache_dir)
        self.enc.src_word_emb = self.enc.embeddings.word_embeddings

        if hparams.share_emb_prj_weight:
            self.word.weight = self.enc.embeddings.word_embeddings.weight
            self.x_logit_scale = (hparams.d_model ** -0.5)
        
        self.loc = nn.Linear(hparams.d_model, 1)
        self.lrb = nn.Sequential(
            nn.Linear(hparams.d_model * 2, hparams.d_model * 2),
            nn.ReLU(),
            nn.Linear(hparams.d_model * 2, 4)
        )
    
    def forward_encoder(self, canvas):
        attention_mask = torch.ones_like(canvas)
        attention_mask = attention_mask.where(canvas!=self.hparams.pad_token_id, torch.tensor(0))
        output=self.enc(canvas, attention_mask=attention_mask)
        return output.last_hidden_state
    
    def training_step(self, batch, batch_idx):
        input_ids, n, n_real = batch['input_ids'], batch['n'], batch['n_real']
        losses = self('losses', input_ids, n, n_real)
        return {**losses, 'log': {**losses}}
    
    def losses(self, seq, n, n_real):
        k = batch_randint(0, n - 1)
        rank = sample_permutation(seq)
        keep = (rank < k.unsqueeze(1))
        canvas, blanks, rest, loc, lb, rb = get_canvas(seq, keep, n)
        loss_loc, loss_word, loss_lrb = self.get_loss(seq, canvas, blanks, rest, loc, lb, rb)
        nll_lb = (loss_loc + loss_word + loss_lrb) * n.float() - (n + 1).float().lgamma()
        return {'loss': nll_lb.sum() / n_real.sum(),
                'loc': loss_loc.mean(),
                'word': loss_word.mean(),
                'lrb': loss_lrb.mean()
                }

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
        target = collect(seq, rest, self.hparams.pad_token_id)
        loss_word = seq_cross_entropy(logits_word, target, self.hparams.pad_token_id)
        loss_word = loss_word.sum(1) / count.float()
        output_word = torch.cat((output_loc, self.enc.src_word_emb(target)), -1)

        logits_lrb = self.lrb(output_word)
        loss_lrb = seq_cross_entropy(logits_lrb, lb * 2 + rb, -3)
        loss_lrb = loss_lrb.sum(1) / count.float()

        return loss_loc, loss_word, loss_lrb
    
    def configure_optimizers(self):
        params_group=[{'params': [param for name, param in self.named_parameters() if 'enc' in name], 'lr': self.hparams.sm_lr},
                {'params': [param for name, param in self.named_parameters() if 'enc' not in name], 'lr': self.hparams.lr}]
        return get_optimizer_and_schedule(params_group, lr=self.hparams.lr)
