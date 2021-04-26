from blm import BLM
from transformers import AutoModel

class PBLM(BLM):
    def __init__(self, hparams):
        self.hparams=hparams
        self.save_hyperparameters()

        self.lrb = nn.Sequential(
            nn.Linear(hparams.d_model * 2, hparams.d_model * 2),
            nn.ReLU(),
            nn.Linear(hparams.d_model * 2, 4)
        )
        self.loc = nn.Linear(hparams.d_model, 1)
        self.enc=AutoModel.from_pretrained(hparams.plm_type)