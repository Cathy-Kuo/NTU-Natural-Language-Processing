# NLP Final Project

## How to train my model
python3.8 train_QA.py
# or with tour data path
python3.8 train_QA.py --data_dir (default=./data/Train_qa_ans.json) --model_out_dir (default=./ckpt/QA)


## How to predict
python3.8 predict_QA.py
# or with tour data path
python3.8 predict_QA.py --data_dir (default=./data/Develop_QA.json) --model_dir (default=./ckpt/QA)