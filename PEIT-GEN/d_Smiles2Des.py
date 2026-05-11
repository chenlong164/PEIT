import argparse
import torch
import numpy as np
from nltk import word_tokenize
from PEIT_GEN import Gen
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, WordpieceTokenizer
from dataset import  SMILESDataset, SMILESProCSV
from torch.utils.data import DataLoader
from calc_property import calculate_property
from rdkit import Chem
import random
import pickle
from tqdm import tqdm
from d_Smiles2Des_sto import generate, BinarySearch
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import nltk
@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, stochastic=False, k=1):
    # test
    print(f"Smiles-to-Des generation in {'stochastic' if stochastic else 'deterministic'} manner with k={k}...")
    model.eval()
    reference, candidate = [], []
    for (prp,text) in tqdm(data_loader):

        # Des.to(device, non_blocking=True)
        # property1 = model.property_embed(prop.unsqueeze(2))  # batch*12*feature
        # properties = torch.cat([model.property_cls.expand(property1.size(0), -1, -1), property1], dim=1)
        # prop_embeds = model.property_encoder(inputs_embeds=properties, return_dict=True).last_hidden_state  # batch*len(=patch**2+1)*feature
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(device)
        # print("Des_input    ",Des_input)
        text_embeds = model.Smiles_encoder.bert(text_input.input_ids[:, 1:],
                                              attention_mask=text_input.attention_mask[:, 1:],
                                              return_dict=True, mode='text').last_hidden_state

        # print("Des_embed    ", Des_embeds)
        product_input = torch.tensor([tokenizer.cls_token_id]).expand(1, 1).to(device)
        values, indices = generate(model, text_embeds, product_input, stochastic=stochastic, k=k)
        product_input = torch.cat([torch.tensor([tokenizer.cls_token_id]).expand(k, 1).to(device), indices.squeeze(0).unsqueeze(-1)], dim=-1)
        current_p = values.squeeze(0)
        final_output = []
        for _ in range(100):
            values, indices = generate(model, text_embeds, product_input, stochastic=stochastic, k=k)
            k2_p = current_p[:, None] + values
            product_input_k2 = torch.cat([product_input.unsqueeze(1).repeat(1, k, 1), indices.unsqueeze(-1)], dim=-1)
            if tokenizer.sep_token_id in indices:
                ends = (indices == tokenizer.sep_token_id).nonzero(as_tuple=False)
                for e in ends:
                    p = k2_p[e[0], e[1]].cpu().item()
                    final_output.append((p, product_input_k2[e[0], e[1]]))
                    k2_p[e[0], e[1]] = -1e5
                if len(final_output) >= k ** 1:
                    break
            current_p, i = torch.topk(k2_p.flatten(), k)
            next_indices = torch.from_numpy(np.array(np.unravel_index(i.cpu().numpy(), k2_p.shape))).T
            product_input = torch.stack([product_input_k2[i[0], i[1]] for i in next_indices], dim=0)

        reference.append(text[0].replace('[CLS]', ''))
        candidate_k = []
        final_output = sorted(final_output, key=lambda x: x[0], reverse=True)[:k]
        for p, sentence in final_output:
            cdd = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(sentence[:-1])).replace('[CLS]', '')
            candidate_k.append(cdd)
        if candidate_k:
            candidate.append(candidate_k[0])
        else:
            print("Warning: candidate_k list is empty for some iteration.")
            candidate.append(float('nan'))
        # candidate.append(candidate_k[0])
        # candidate.append(random.choice(candidate_k))
    for index, value in enumerate(candidate):
        # 这里可以添加你想要对每一行执行的操作
        print(f"行索引: {index}, 行值: {value}")
    return reference, candidate
@torch.no_grad()
def metric_NLP(candidate):
    output_filename = "smiles_description.csv"
    df = pd.read_csv(args.input_file)
    SMILES_list = df["SMILES"]
    Des_list = df["description"]
    Predict = candidate
    df = pd.DataFrame({
        "SMILES":SMILES_list,
        "Description":Des_list,
        "Predict":candidate
    }
    )
    df.to_csv(output_filename,index=False)
    # df.to_csv(output_filename, index=False)
    # df_1.to_csv(output_filename1, index=False)
    print(f"The SMILES and generated descriptions have been saved to {output_filename}")
    # df.to_csv("output1.txt", index=False, sep='\t', header=True)
    # 可以在这里返回累计的 BLEU 分数或其他统计信息
# def metric_NLP(candidate):
#     Ground_trueth = []
#     Ground_trueth = pd.read_csv(args.input_file)["description"]
#     for index, value in enumerate(candidate):
#         GT = Ground_trueth[index]
#         GT = nltk.tokenize(GT)
#         Pred = nltk.tokenize(value)
#         sb_total = 0
#         bleu = sentence_bleu(GT,Pred,weights=(1, 0, 0, 0))
#         sb_total += bleu
#         print("bleu-1 score:",bleu)



def main(args, config):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = random.randint(0, 1000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # ### Dataset ### #
    print("Creating dataset")
    # dataset_test = SMILESDataset_pretrain(args.input_file)
    dataset_test = SMILESProCSV(args.input_file)
    test_loader = DataLoader(dataset_test, batch_size=1, pin_memory=True, drop_last=False)
    tokenizerSD = BertTokenizer.from_pretrained("sci_bert")
    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)

    # === Model === #
    print("Creating model")
    model = Gen(config=config, tokenizerSP=tokenizer,tokenizerSD=tokenizerSD, no_train=True)
    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['state_dict']

        for key in list(state_dict.keys()):
            if 'word_embeddings' in key and 'property_encoder' in key:
                del state_dict[key]
            if 'queue' in key:
                del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)
    model = model.to(device)

    print("=" * 50)
    r_test, c_test = evaluate(model, test_loader, tokenizerSD, device, stochastic=args.stochastic, k=args.k)
    print(r_test)
    print("=" * 50)
    print(c_test)
    print("=" * 50)
    metric_NLP(c_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint', default='./modelpth/checkpoint_SPMM_001.ckpt')
    parser.add_argument('--checkpoint', default='./modelpth/checkpoint_SDPFusion_epoch=479.ckpt')
    # parser.add_argument('--vocab_filename', default='./vocab_bpe_300.txt')
    parser.add_argument('--vocab_filename', default='./sci_bert/vocab.txt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--input_file', default='1111.csv', type=str)
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--stochastic', default=False, type=bool)
    args = parser.parse_args()

    config = {
        'embed_dim': 256,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config.json',
    }
    main(args, config)
