import argparse
import torch
import numpy as np
from PEIT_GEN import Gen
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, WordpieceTokenizer
from dataset import SMILESDataset_pretrain,SMILESProCSV
from torch.utils.data import DataLoader
import random
import pickle
from sklearn.metrics import r2_score
from tqdm import tqdm


def generate(model, prop_input, text_embeds, text_atts):
    prop_embeds = model.property_encoder(inputs_embeds=prop_input, return_dict=True).last_hidden_state
    prob_atts = torch.ones(prop_input.size()[:-1], dtype=torch.long).to(prop_input.device)
    token_output = model.text_encoder.bert(encoder_embeds=prop_embeds,
                                           attention_mask=prob_atts,
                                           encoder_hidden_states=text_embeds,
                                           encoder_attention_mask=text_atts,
                                           return_dict=True,
                                           is_decoder=True,
                                           mode='fusion',
                                           ).last_hidden_state
    pred = model.property_mtr_head(token_output).squeeze(-1)[:, -1]
    return pred.unsqueeze(1)


@torch.no_grad()
def pv_generate(model, data_loader):
    # test
    with open('normalize.pkl', 'rb') as w:
        mean, std = pickle.load(w)
    device = model.device
    tokenizer = model.tokenizer
    model.eval()
    print("SMILES-to-PV generation...")
    # convert list of string to dataloader
    if isinstance(data_loader, list):
        gather = []
        text_input = tokenizer(data_loader, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(device)
        text_embeds = model.text_encoder.bert(text_input.input_ids[:, 1:], attention_mask=text_input.attention_mask[:, 1:],
                                              return_dict=True, mode='text').last_hidden_state
        prop_input = model.property_cls.expand(len(data_loader), -1, -1)
        prediction = []
        for _ in range(53):
            output = generate(model, prop_input, text_embeds, text_input.attention_mask[:, 1:])
            prediction.append(output)
            output = model.property_embed(output.unsqueeze(2))
            prop_input = torch.cat([prop_input, output], dim=1)

        prediction = torch.stack(prediction, dim=-1)
        for i in range(len(data_loader)):
            gather.append(prediction[i].cpu()*std + mean)
        return gather

    reference, candidate = [], []
    for (prop, text) in tqdm(data_loader):
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(device)
        text_embeds = model.text_encoder.bert(text_input.input_ids[:, 1:], attention_mask=text_input.attention_mask[:, 1:],
                                              return_dict=True, mode='text').last_hidden_state
        prop_input = model.property_cls.expand(len(text), -1, -1)
        prediction = []
        for _ in range(53):
            output = generate(model, prop_input, text_embeds, text_input.attention_mask[:, 1:])
            prediction.append(output)
            output = model.property_embed(output.unsqueeze(2))
            prop_input = torch.cat([prop_input, output], dim=1)

        prediction = torch.stack(prediction, dim=-1)
        for i in range(prop.size(0)):
            reference.append(prop[i].cpu())
            candidate.append(prediction[i].cpu())
        with open('normalize.pkl', 'rb') as w:
            norm = pickle.load(w)
        mean, std = norm
        mse = []
        n_mse = []
        rs, cs = [], []
        for i in range(len(candidate)):
            r = (reference[i] * std) + mean
            c = (candidate[i] * std) + mean
            rs.append(r)
            cs.append(c)
    print('SMILES-to-PV generation done')
    # for index, value in enumerate(candidate):
    #     # 这里可以添加你想要对每一行执行的操作
    #     print(f"行索引: {index}, 行值: {value[0][0]}")
    return rs, cs

def main(args, config):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = random.randint(0, 1000)
    print('seed:', seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # === Dataset === #
    print("Creating dataset")
    dataset_test = SMILESProCSV(args.input_file)
    test_loader = DataLoader(dataset_test, batch_size=config['batch_size_test'], pin_memory=True, drop_last=False)

    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)
    tokenizer1 = BertTokenizer.from_pretrained("sci_bert")
    # === Model === #
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
    with open('normalize.pkl', 'rb') as w:
        norm = pickle.load(w)
    mean, std = norm

    print("=" * 50)
    r_test, c_test = pv_generate(model, test_loader)
    print(r_test)
    print(c_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./modelpth/checkpoint_SDPFusion_epoch=6-v1.ckpt')
    # parser.add_argument('--checkpoint', default='./modelpth/checkpoint_SPMM-001.ckpt')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300.txt')
    parser.add_argument('--input_file', default='./data/2_PV2SMILES/pubchem_1k_unseen2.csv')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    config = {
        'embed_dim': 256,
        'batch_size_test': 64,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
    }
    main(args, config)
