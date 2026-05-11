import argparse
import csv
import random

import numpy as np
import pandas as pd
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, WordpieceTokenizer
from PEIT_GEN import Gen
from dataset import SMILESProCSV
from d_Smiles2Des import evaluate
from d_smiles2pv import pv_generate
attributes = [
    "BalabanJ",
    "BertzCT",
    "Chi0",
    "Chi0n",
    "Chi0v",
    "Chi1",
    "Chi1n",
    "Chi1v",
    "Chi2n",
    "Chi2v",
    "Chi3n",
    "Chi3v",
    "Chi4n",
    "Chi4v",
    "ExactMolWt",
    "FpDensityMorgan1",
    "FpDensityMorgan2",
    "FpDensityMorgan3",
    "FractionCSP3",
    "HallKierAlpha",
    "HeavyAtomCount",
    "HeavyAtomMolWt",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA",
    "MaxAbsEStateIndex",
    "MaxEStateIndex",
    "MinAbsEStateIndex",
    "MinEStateIndex",
    "MolLogP",
    "MolMR",
    "MolWt",
    "NHOHCount",
    "NOCount",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRadicalElectrons",
    "NumRotatableBonds",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
    "NumValenceElectrons",
    "RingCount",
    "TPSA",
    "QED"
]
def main(args, config):
    device = torch.device(args.device)
    seed = random.randint(0, 1000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    print("Creating dataset")
    dataset_test = SMILESProCSV(args.input_file)
    test_loader = DataLoader(dataset_test, batch_size=1, pin_memory=True, drop_last=False)
    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)
    tokenizer1 = BertTokenizer.from_pretrained("sci_bert")
    print("Creating model")
    model = Gen(config=config, tokenizerSP=tokenizer, tokenizerSD=tokenizer1, no_train=True)
    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['state_dict']

        for key in list(state_dict.keys()):
            if 'queue' in key:
                del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)
    model = model.to(device)

    print("=" * 50)
    smiles_list =  pd.read_csv(args.input_file)["SMILES"]
    _, Des_list = evaluate(model, test_loader, tokenizer1, device, stochastic=args.stochastic, k=args.k)
    _, pv_list = pv_generate(model, test_loader)

    pv_dict = {attribute: [] for attribute in attributes}
    for value in pv_list:
        for i, attribute in enumerate(attributes):
            pv_dict[attribute].append(float(value[0][i]))
    print(pv_dict)
    df = pd.DataFrame({
        "SMILES": smiles_list,
        "Description": Des_list,
        **pv_dict
    })
    df.to_csv(args.output_file,index=False)
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='../modelpth/checkpoint_PEIT_epoch.ckpt')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300.txt')
    parser.add_argument('--input_file', default='../Template_Generate/data/10000.csv')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_file', default='../Template_Generate/100.csv')
    parser.add_argument('--stochastic', default=False, type=bool)
    parser.add_argument('--k', default=1, type=int)
    args = parser.parse_args()

    config = {
        'embed_dim': 256,
        'batch_size_test': 64,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
    }
    main(args, config)
