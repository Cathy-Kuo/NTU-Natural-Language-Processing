import json
import torch
torch.cuda.is_available()
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

from tqdm import trange

from pathlib import Path
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from dataset import QAgpt
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import random
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel, GPT2TokenizerFast, BertTokenizerFast

torch.cuda.empty_cache()

output_dir = Path("ckiplab/gpt2-base-chinese")
model = GPT2DoubleHeadsModel.from_pretrained(output_dir)
tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-macbert-base')

str_path = "./data/Train_qa_ans.json"
path = Path(str_path)
qa_data = json.loads(path.read_text())

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
    #d['text'] = d['text'].replace("。", "")
    #d['text'] = d['text'].replace("，", "")
    d['text'] = d['text'].replace("‧", "")
    d['text'] = d['text'].replace("！", "")

#train_data, val_data = train_test_split(qa_data, random_state=77, train_size=0.95)
train_data = qa_data

str_path = "./data/Develop_QA.json"
path = Path(str_path)
qa_dev = json.loads(path.read_text())

for d in qa_dev:
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
    #d['text'] = d['text'].replace("。", "")
    #d['text'] = d['text'].replace("，", "")
    d['text'] = d['text'].replace("‧", "")
    d['text'] = d['text'].replace("！", "")
    
f = open('./data/ans.txt')
text = f.readlines()
val_data = []
for i in range(len(text)):
    qa_dev[i]['answer'] = text[i][0]
    val_data.append(qa_dev[i])

for i in range(len(train_data)):
    if train_data[i]['id']==91:
        train_data[i]['answer']='C'
        
###not sure padding and start tensor end tensor
def collate(samples) :
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    masks_tensors = [s[2] for s in samples]
    label_tensor = torch.stack([s[3] for s in samples])

    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    masks_tensors = pad_sequence(masks_tensors, batch_first=True)

    return tokens_tensors, segments_tensors, masks_tensors, label_tensor


trainset = QAgpt(data=train_data, tokenizer=tokenizer)
valset = QAgpt(data=val_data, tokenizer=tokenizer)

BATCH_SIZE = 1
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=True)
valloader = DataLoader(valset, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=False)

def cal_acc(label_pred, label_true):
    c = 0
    for i in range(len(label_pred)):
        if (label_pred[i] == label_true[i]):
            c += 1
    return c
#gradient accumulation
from transformers import AdamW, Adafactor

use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()
print('gpu:',use_gpu)

EPOCHS = 38

learning_rate = 1e-4
def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.5 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

optimizer = Adafactor(model.parameters(), lr=learning_rate, relative_step=False)

accumulation_steps = 4
    

for epoch in tqdm(range(EPOCHS)):
    
    running_loss = 0.0
    acc = 0
    model.train()
    model.zero_grad()                                   # Reset gradients tensors
    for i, data in tqdm(enumerate(trainloader)):
        if i%50==0:
            print('i:',i)
        if use_gpu:
            tokens_tensors, segments_tensors, masks_tensors, label_tensors = [t.cuda() for t in data]
        else:
            tokens_tensors, segments_tensors, masks_tensors, label_tensors = [t for t in data]
        
        outputs = model(input_ids=tokens_tensors, 
                        token_type_ids=segments_tensors, 
                        attention_mask=masks_tensors, 
                        mc_labels=label_tensors)
        
        loss = outputs.mc_loss                          # Compute loss function
        loss = loss / accumulation_steps                # Normalize our loss (if averaged)
        loss.backward()                                 # Backward pass
        if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
            optimizer.step()                            # Now we can do an optimizer step
            optimizer.zero_grad()                           # Reset gradients tensors     
        
        label_preds = torch.argmax(outputs.mc_logits, dim=1)
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
                        mc_labels=label_tensors)

            loss = outputs.mc_loss

            label_preds = torch.argmax(outputs.mc_logits, dim=1)
            acc += cal_acc(label_preds, label_tensors)

            running_loss += loss.item()

        print('epoch: ', epoch + 1,  ', validation loss: ',running_loss, ', acc: ', acc/len(val_data))
