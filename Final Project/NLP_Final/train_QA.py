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

from transformers import BertTokenizerFast, BertTokenizer, BertForMultipleChoice, AdamW
from transformers import AutoConfig, AutoModelForMultipleChoice, AutoTokenizer

class MultiChoiceDataset1(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.len = len(self.data)
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        ans_dict = {'A':0, 'B':1, 'C':2, 'Ａ':0, 'Ｂ':1, 'Ｃ':2, 'A ':0, 'B ':1, 'C ':2}
        
        q = self.data[idx]['question']['stem']
        choices = self.data[idx]['question']['choices']
        label = ans_dict[self.data[idx]['answer']]
        
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
        label_tensor = torch.tensor(label)
        
        return (tokens_tensor, segments_tensor, mask_tensor, label_tensor)
    
    def __len__(self):
        return self.len

def collate(samples) :
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    masks_tensors = [s[2] for s in samples]
    label_tensor = torch.stack([s[3] for s in samples])

    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    masks_tensors = pad_sequence(masks_tensors, batch_first=True)

    return tokens_tensors, segments_tensors, masks_tensors, label_tensor

def cal_acc(label_pred, label_true):
    c = 0
    for i in range(len(label_pred)):
        if (label_pred[i] == label_true[i]):
            c += 1
    return c

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.5 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def main(args):
    model = AutoModelForMultipleChoice.from_pretrained("hfl/chinese-macbert-base")
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
 
    data_path = args.data_dir
    qa_data = json.loads(data_path.read_text())

    for d in qa_data:
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
    
    for i in range(len(qa_data)):
        if qa_data[i]['id']==91:
            qa_data[i]['answer']='C'

    train_data, val_data = train_test_split(qa_data, random_state=77, train_size=0.8)
    
    trainset = MultiChoiceDataset1(data=train_data, tokenizer=tokenizer)
    valset = MultiChoiceDataset1(data=val_data, tokenizer=tokenizer)

    BATCH_SIZE = 1
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=True)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=False)
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

    EPOCHS = 30
    learning_rate = 6e-6
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    accumulation_steps = 4

    for epoch in tqdm(range(EPOCHS)):

        running_loss = 0.0
        acc = 0
        model.train()
        model.zero_grad()                              
        for i, data in tqdm(enumerate(trainloader)):
            if use_gpu:
                tokens_tensors, segments_tensors, masks_tensors, label_tensors = [t.cuda() for t in data]
            else:
                tokens_tensors, segments_tensors, masks_tensors, label_tensors = [t for t in data]

            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors, 
                            labels=label_tensors)

            loss = outputs[0]                          
            loss = loss / accumulation_steps          
            loss.backward()                        
            if (i+1) % accumulation_steps == 0:  
                optimizer.step()              
                optimizer.zero_grad()   

            label_preds = torch.argmax(outputs[1], dim=1)
            acc += cal_acc(label_preds, label_tensors)

            running_loss += loss.item()

        print('epoch: ', epoch + 1,  ', loss: ',running_loss, ', acc: ', acc/len(train_data))

        running_loss = 0.0
        acc = 0
        model.eval()
        with torch.no_grad():
            for data in valloader:
                if use_gpu:
                    tokens_tensors, segments_tensors, masks_tensors, label_tensors = [t.cuda() for t in data]
                else:
                    tokens_tensors, segments_tensors, masks_tensors, label_tensors = [t for t in data]


                outputs = model(input_ids=tokens_tensors, 
                                token_type_ids=segments_tensors, 
                                attention_mask=masks_tensors, 
                                labels=label_tensors)

                loss = outputs[0]

                label_preds = torch.argmax(outputs[1], dim=1)
                acc += cal_acc(label_preds, label_tensors)

                running_loss += loss.item()

            print('epoch: ', epoch + 1,  ', validation loss: ',running_loss, ', acc: ', acc/len(val_data))
    
    output_dir = args.model_out_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_to_save = model.module if hasattr(model, 'module') else model 
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/Train_qa_ans.json",
    )
    parser.add_argument(
        "--model_out_dir",
        type=Path,
        help="Directory to the model.",
        default="./ckpt/QA",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.model_out_dir.mkdir(parents=True, exist_ok=True)
    main(args)
