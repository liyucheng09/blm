from lyc.train import get_args, TrainingArgs
from dataclasses import dataclass
from lyc.utils import (get_tokenizer,
                       get_pl_callbacks)
from lyc.data import (get_tokenized_ds,
                      processor,
                      get_dataloader)
from models.blm_pretrained import PBLM
import pytorch_lightning as pl
import torch

@dataclass
class PblmArgs(TrainingArgs):
    is_zh : bool = True
    ds_name : str = 'train'
    d_model : int = 768
    prefix : str = 'v0.1'

if __name__ == '__main__':
    args = get_args(PblmArgs)
    args.gpus = torch.cuda.device_count()
    args.multigpu = torch.cuda.device_count() > 1

    tokenizer=get_tokenizer(args.tokenizer_name_or_path, 
            is_zh=args.is_zh, max_length = args.max_sent_len, min_sent_length = args.min_sent_len)
    ds = get_tokenized_ds(args.dataset_scripts, 
        args.train_data, tokenizer, max_length = args.max_sent_len, 
        min_sent_length=args.min_sent_len, shuffle=True, tokenize_func='no_padding')
    train_ds=ds[args.ds_name]

    processor.block_size = 512
    processor.tokenizer = tokenizer
    train_ds = train_ds.map(
        processor.lm_group_texts,
        batched=True
    )
    train_ds = train_ds.map(
        processor.get_true_length,
        batched=True
    )
    train_dl=get_dataloader(train_ds, batch_size=args.batch_size, 
        cols=['input_ids', 'n', 'n_real'])

    args.vocab_size = tokenizer.vocab_size
    args.pad_token_id = tokenizer.pad_token_id
    model = PBLM(args)
    callbacks = get_pl_callbacks(args)

    trainer = pl.Trainer(
        max_epochs = args.epoches, 
        gpus = args.gpus,
        distributed_backend = 'ddp' if args.multigpu else None,
        log_every_n_steps = args.log_steps,
        default_root_dir = args.root_dir,
        callbacks = callbacks['checkpoint'],
        # limit_train_batches=10x
    )

    trainer.fit(model, train_dataloader=train_dl)
