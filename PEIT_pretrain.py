from torch.utils.data import DataLoader
from dataset import SMILESDescriptionProperties
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import torch.distributed
from MySDPFusion import SDPFusionModel
import argparse
from pathlib import Path
from transformers import BertTokenizer, WordpieceTokenizer

def main(args, config):
    ngpu=1
    # data
    print("Creating dataset")
    dataset = SMILESDescriptionProperties(args.data_path)
    print('#data:', len(dataset), torch.cuda.is_available())
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=8, shuffle=False, pin_memory=True, drop_last=True)
    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)
    tokenizerSD = BertTokenizer.from_pretrained("sci_bert")

    # model
    model = SDPFusionModel(config=config, tokenizerSP=tokenizer,tokenizerSD=tokenizerSD, loader_len=len(data_loader) // torch.cuda.device_count())

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} is unfreezed and will be trained.")
    # training
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.output_dir, filename='checkpoint_SDPFusion_{epoch}',
                                                       #save_top_k=2,
                                                       #monitor='epoch',
                                                       #every_n_epochs=1,
                                                       every_n_train_steps=10000,
                                                       )
    trainer = pl.Trainer(accelerator='gpu', devices=1, precision='16-mixed', max_epochs=config['schedular']['epochs'],
                         callbacks=[checkpoint_callback], strategy=DDPStrategy(find_unused_parameters=True), limit_val_batches=0.)
    trainer.fit(model, data_loader, None, ckpt_path=args.checkpoint if args.checkpoint else None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--data_path', default='./data/SMILES_Des.csv')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='./Pretrain')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300.txt')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    pretrain_config = {
        'property_width': 768,
        'embed_dim': 256,
        'batch_size': 16,
        'temp': 0.07,
        'mlm_probability': 0.15,
        'queue_size': 36864, # 24576,
        'momentum': 0.995,
        'alpha': 0.4,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
        'schedular': {'sched': 'cosine', 'lr': 5e-5, 'epochs': 30, 'min_lr': 1e-5,
                      'decay_rate': 1, 'warmup_lr': 5e-5, 'warmup_epochs': 20, 'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': 5e-5, 'weight_decay': 0.02}
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, pretrain_config)
