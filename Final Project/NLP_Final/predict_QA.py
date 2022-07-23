import json
import torch
torch.cuda.is_available()
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import os

import math

from tqdm import trange
from tqdm.notebook import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import random
import csv

from transformers import BertTokenizerFast, BertTokenizer, BertForMultipleChoice, AdamW
from transformers import AutoConfig, AutoModelForMultipleChoice, AutoTokenizer

class MultiChoiceDataset2(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.len = len(self.data)
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        
        q = self.data[idx]['question']['stem']
        choices = self.data[idx]['question']['choices']
        
        text = self.data[idx]['text']
        t = self.tokenizer(text,max_length=490,truncation="only_first", stride=256,return_overflowing_tokens=True)
        tqa = self.tokenizer.encode(q + choices[0]['text'] + choices[1]['text'] + choices[2]['text'])
        overflow = t.overflow_to_sample_mapping
        m=0
        idx = 0
        for i in range(len(overflow)):
            if len(set(t['input_ids'][i])&set(tqa))>m:
                m=len(set(t['input_ids'][i])&set(tqa))
                idx=i
                
        text_b = []
        for c in choices:
            text_b.append(q + c['text'])
        
        text = self.tokenizer.decode(t['input_ids'][idx], skip_special_tokens=True).replace(" ", "")
        
        t = self.tokenizer(text=[text,text,text],text_pair=text_b, add_special_tokens=True,max_length=512,truncation='only_first', return_tensors='pt', padding=True)
        tokens_tensor = t['input_ids']
        segments_tensor = t['token_type_ids']
        mask_tensor = t['attention_mask']
        
        return (tokens_tensor, segments_tensor, mask_tensor)
    
    def __len__(self):
        return self.len

def collate1(samples) :
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    masks_tensors = [s[2] for s in samples]

    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    masks_tensors = pad_sequence(masks_tensors, batch_first=True)

    return tokens_tensors, segments_tensors, masks_tensors


def main(args):
    modelDir = args.model_dir
    model = AutoModelForMultipleChoice.from_pretrained(modelDir)
    tokenizer = AutoTokenizer.from_pretrained(modelDir)
    
    data_path = args.data_dir
    dev_data = json.loads(data_path.read_text())

    for d in dev_data:
        d['text'] = d['text'].replace("醫師：", "")
        d['text'] = d['text'].replace("護理師：", "")
        d['text'] = d['text'].replace("民眾：", "")
        d['text'] = d['text'].replace("喔", "")
        d['text'] = d['text'].replace("嘿", "")
        d['text'] = d['text'].replace("嗯", "")
        d['text'] = d['text'].replace("恩。", "")
        d['text'] = d['text'].replace("恩", "")
        d['text'] = d['text'].replace("阿", "")
        d['text'] = d['text'].replace("啊", "")
        d['text'] = d['text'].replace("啦", "")
        d['text'] = d['text'].replace("欸", "")
        d['text'] = d['text'].replace("哼", "")
        d['text'] = d['text'].replace("呵", "")
        d['text'] = d['text'].replace("的", "")
        d['text'] = d['text'].replace("了", "")
        d['text'] = d['text'].replace("", "")
        d['text'] = d['text'].replace("。。", "。")
        d['text'] = d['text'].replace("。，", "，")
        d['text'] = d['text'].replace("？，", "？")
        d['text'] = d['text'].replace("……", "")
        d['text'] = d['text'].replace("，，", "，")
        d['text'] = d['text'].replace("⋯⋯", "")
        d['text'] = d['text'].replace("‧", "")
        d['text'] = d['text'].replace("！", "")

    devset = MultiChoiceDataset2(data=dev_data, tokenizer=tokenizer)
    BATCH_SIZE = 1
    devloader = DataLoader(devset, batch_size=BATCH_SIZE, collate_fn=collate1, shuffle=False)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

    ans_dict = {0:'A', 1:'B', 2:'C'}
    ans = []
    ids = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(devloader):
            if use_gpu:
                tokens_tensors, segments_tensors, masks_tensors = [t.cuda() for t in data]
            else:
                tokens_tensors, segments_tensors, masks_tensors = [t for t in data]


            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            label_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            ans.append(ans_dict[label_preds[0]])
        ids.append(dev_data[i]['id'])

    with open('qa.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'answer'])
        for i in range(len(ids)):
            writer.writerow([ids[i], ans[i]])

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/Develop_QA.json",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        help="Directory to the model.",
        default="./ckpt/QA",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
