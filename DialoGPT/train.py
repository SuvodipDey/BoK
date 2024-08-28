from datasets import Dataset, DatasetDict, load_dataset
import transformers
from transformers import AutoTokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, default_data_collator, EarlyStoppingCallback
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import logging
import time
import argparse
import shutil
from create_data import get_data
import os
import pandas as pd
import yake

#----------------------------

"""
Training script:-

1. With BoK loss
python train.py -path=<model_dir> -src_file=train.py -dt=dd/pc -key

2. With BoW loss
python train.py -path=<model_dir> -src_file=train.py -dt=dd/pc -key -all

3. Basic Model
python train.py -path=<model_dir> -src_file=train.py -dt=dd/pc

"""
#----------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-path','--path', help='path of the model directiory', required=True)
parser.add_argument('-src_file','--src_file', help='path of the source file', required=True)
parser.add_argument('-dt','--dt', help='Name of dataset: dd (DailyDialg), pc (PersonaChat)', required=True, choices=['dd', 'pc'])
parser.add_argument('-model','--model', help='T5 model name', default='large', choices=['small', 'medium', 'large'])
parser.add_argument('-key','--key', help='Use keywords', default=False, action='store_true')
parser.add_argument('-all','--all', help='Use all words as keyword', default=False, action='store_true')
parser.add_argument('-pos','--pos', help='Add POS keywords', default=False, action='store_true')
parser.add_argument('-bow_len','--bow_len', help='Use bow loss', type=int, required=False, default=8)
parser.add_argument('-test','--test', help='Test Run', default=False, action='store_true')
parser.add_argument('-epochs','--epochs', help='Number of epochs', type=int, required=False, default=10)
parser.add_argument('-batch','--batch', help='Batch size', type=int, required=False, default=16)
parser.add_argument('-lr','--lr', help='Learning rate', type=float, required=False, default=5e-5)
parser.add_argument('-topn','--topn', help='top-n keywords', type=int, required=False, default=20)
parser.add_argument('-ng','--ng', help='max_ngram_size', type=int, required=False, default=1)
parser.add_argument('-m','--m', help='Weight of bow loss', type=float, required=False, default=0.1)

args = vars(parser.parse_args())
model_dir = args['path']
src_file = args['src_file']
dataset = args['dt']
model_name = args['model']
use_key = args['key']
add_pos = args['pos']
test_run = args['test']
num_train_epochs = args['epochs']
batch_size = args['batch']
learning_rate = args['lr']
topn = args['topn']
ng_size = args['ng']
lst_label_names = ["labels"]
bow_wt = args['m']
all_key = args['all']
bow_len = args['bow_len']

if(all_key and not use_key):
    print("if all-key is True, key must be True")
    exit(0)
    
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print(f"Using GPU!")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
max_len = 512
max_len_response = 64
max_bow = bow_len
if(all_key):
    max_bow = max_len_response
    
if(use_key):
    lst_label_names.append("key_ids")
        
if(os.path.isdir(model_dir)):
    print(f"Model Directory {model_dir} exists.")
    exit(0)
else:
    os.mkdir(model_dir)
    shutil.copy(src_file, model_dir)
    print(f"Model directory {model_dir} created.")
    
MODEL_CKPT = "microsoft/DialoGPT-small"    
hidden_size = 768   
if(model_name=="medium"):
    MODEL_CKPT = "microsoft/DialoGPT-medium"
    hidden_size = 1024
elif(model_name=="large"):
    MODEL_CKPT = "microsoft/DialoGPT-large"
    hidden_size = 1280
    
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)

new_tokens = {'eou': tokenizer.eos_token, 'rep': '<rep>'}
if(dataset=="pc"):
    new_tokens['knlg'] = tokenizer.eos_token
if(use_key):
    new_tokens['nok'] = '<nok>'

tokenizer.add_tokens([new_tokens['rep']], special_tokens=True)
if(use_key):
    tokenizer.add_tokens([new_tokens['nok']], special_tokens=True)

log_dir = os.path.join(model_dir, "logs")
rep_token_id = tokenizer.convert_tokens_to_ids(new_tokens['rep'])
print("Tokenizer loaded.")

#Setting log file
log_file = os.path.join(model_dir, 'log.txt')
logging.basicConfig(filename=log_file, filemode='a', 
                    format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logging.Formatter.converter = time.gmtime
logger = logging.getLogger(__name__)
transformers.utils.logging.set_verbosity(logging.WARNING) 

logger.info(args)
logger.info("-"*30)
logger.info(f"dataset: {dataset}")
logger.info(f"new_tokens: {new_tokens}")
logger.info(f"max_len={max_len} and max_len_response={max_len_response}")
if(use_key):
    logger.info(f"max_bow={max_bow}")
logger.info(f"rep token id: {rep_token_id}")
logger.info("-"*30)
                    
#----------------------------
## Yake
language = "en"
max_ngram_size = ng_size
deduplication_thresold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = topn
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)

#----------------------------
        
def trim_context(st, lst, size):
    str_match = f"{new_tokens['eou']}"
    try:
        ind = st.index(str_match)
    except :
        return st
    if(len(lst)>(size+1)):
        trm_st = st[ind+len(str_match):]
        trm_lst = tokenizer.tokenize(trm_st)
        st = trim_context(trm_st, trm_lst, size)
        return st
    else:
        return st

def prepare_features(df):
    df_input = df["input"].lower()
    response = df["output"].lower()
    trm_input = trim_context(df_input, tokenizer.tokenize(df_input), max_len)
    
    input_ids = tokenizer.encode(trm_input + new_tokens['rep'])
    resp_ids = tokenizer.encode(response, truncation=True, max_length=max_len_response)
    resp_ids.append(tokenizer.eos_token_id)
    
    n = len(input_ids)
    label = [-100]*n
    label.extend(resp_ids)
    input_ids.extend(resp_ids)
    output = {}
    output["input_ids"] = input_ids
    output["labels"] = label
    
    if(use_key):
        tok_key = tokenizer.encode(df["key_words"])
        n = len(tok_key)
        if (n>max_bow):
            tok_key = tok_key[:max_bow]
        else:
            for i in range(max_bow-n):
                tok_key.append(-100)
        output["key_ids"] = tok_key
    return output

#----------------------------                 

## Data Processing
print("Tokenizing Data ...")
logger.info("Tokenizing Data ...")
df = get_data(dataset, new_tokens, use_key, all_key, custom_kw_extractor, add_pos, tokenizer, test_run)

## Log data snippet for verification
print(df)
logger.info(df)
logger.info("-"*30)
logger.info("Data Snippet:-")
idx=1
logger.info(df["test"]["input"][idx])
logger.info(df["test"]["output"][idx])
if(use_key):
    logger.info(df["test"]["key_words"][idx])
logger.info("-"*30)
logger.info(df["test"]["input"][idx+1])
logger.info(df["test"]["output"][idx+1])
if(use_key):
    logger.info(df["test"]["key_words"][idx+1])
logger.info("-"*30)
logger.info(df["test"]["input"][idx+2])
logger.info(df["test"]["output"][idx+2])
if(use_key):
    logger.info(df["test"]["key_words"][idx+2])
logger.info("-"*30)
logger.info(df["test"]["input"][idx+3])
logger.info(df["test"]["output"][idx+3])
if(use_key):
    logger.info(df["test"]["key_words"][idx+3])
logger.info("-"*30)
logger.info(df["test"]["input"][idx+4])
logger.info(df["test"]["output"][idx+4])
if(use_key):
    logger.info(df["test"]["key_words"][idx+4])
logger.info("-"*30)
logger.info("-"*30)

## Tokenize data
tokenized_datasets = df.map(prepare_features, remove_columns=df["train"].column_names)

## Log data snippet for verification
logger.info("Tokenized Data Snippet:-")
idx=1
logger.info(f"input_ids: {tokenized_datasets['test']['input_ids'][idx]}")
logger.info(f"input_ids: {tokenizer.decode(tokenized_datasets['test']['input_ids'][idx], skip_special_tokens=False)}")
lbl = tokenized_datasets["test"]["labels"][idx]
lbl = [v if v!=-100 else tokenizer.eos_token_id for v in lbl]
logger.info(f"labels: {tokenizer.decode(lbl, skip_special_tokens=True)}")
logger.info(f"labels ids: {tokenized_datasets['test']['labels'][idx]}")
if(use_key):
    key_lbl = tokenized_datasets['test']['key_ids'][idx]
    key_lbl = [v if v!=-100 else tokenizer.eos_token_id for v in key_lbl]
    logger.info(f"key_ids: {tokenizer.decode(key_lbl, skip_special_tokens=False)}")
    logger.info(f"key_ids: {tokenized_datasets['test']['key_ids'][idx]}")
logger.info("-"*30)

idx=idx+1
logger.info(f"input_ids: {tokenized_datasets['test']['input_ids'][idx]}")
logger.info(f"input_ids: {tokenizer.decode(tokenized_datasets['test']['input_ids'][idx], skip_special_tokens=False)}")
lbl = tokenized_datasets["test"]["labels"][idx]
lbl = [v if v!=-100 else tokenizer.eos_token_id for v in lbl]
logger.info(f"labels: {tokenizer.decode(lbl, skip_special_tokens=True)}")
logger.info(f"labels ids: {tokenized_datasets['test']['labels'][idx]}")
if(use_key):
    key_lbl = tokenized_datasets['test']['key_ids'][idx]
    key_lbl = [v if v!=-100 else tokenizer.eos_token_id for v in key_lbl]
    logger.info(f"key_ids: {tokenizer.decode(key_lbl, skip_special_tokens=False)}")
    logger.info(f"key_ids: {tokenized_datasets['test']['key_ids'][idx]}")
logger.info("-"*30)

idx=idx+1
logger.info(f"input_ids: {tokenized_datasets['test']['input_ids'][idx]}")
logger.info(f"input_ids: {tokenizer.decode(tokenized_datasets['test']['input_ids'][idx], skip_special_tokens=False)}")
lbl = tokenized_datasets["test"]["labels"][idx]
lbl = [v if v!=-100 else tokenizer.eos_token_id for v in lbl]
logger.info(f"labels: {tokenizer.decode(lbl, skip_special_tokens=True)}")
logger.info(f"labels ids: {tokenized_datasets['test']['labels'][idx]}")
if(use_key):
    key_lbl = tokenized_datasets['test']['key_ids'][idx]
    key_lbl = [v if v!=-100 else tokenizer.eos_token_id for v in key_lbl]
    logger.info(f"key_ids: {tokenizer.decode(key_lbl, skip_special_tokens=False)}")
    logger.info(f"key_ids: {tokenized_datasets['test']['key_ids'][idx]}")
logger.info("-"*30)

idx=idx+1
logger.info(f"input_ids: {tokenized_datasets['test']['input_ids'][idx]}")
logger.info(f"input_ids: {tokenizer.decode(tokenized_datasets['test']['input_ids'][idx], skip_special_tokens=False)}")
lbl = tokenized_datasets["test"]["labels"][idx]
lbl = [v if v!=-100 else tokenizer.eos_token_id for v in lbl]
logger.info(f"labels: {tokenizer.decode(lbl, skip_special_tokens=True)}")
logger.info(f"labels ids: {tokenized_datasets['test']['labels'][idx]}")
if(use_key):
    key_lbl = tokenized_datasets['test']['key_ids'][idx]
    key_lbl = [v if v!=-100 else tokenizer.eos_token_id for v in key_lbl]
    logger.info(f"key_ids: {tokenizer.decode(key_lbl, skip_special_tokens=False)}")
    logger.info(f"key_ids: {tokenized_datasets['test']['key_ids'][idx]}")
logger.info("-"*30)

idx=idx+1
logger.info(f"input_ids: {tokenized_datasets['test']['input_ids'][idx]}")
logger.info(f"input_ids: {tokenizer.decode(tokenized_datasets['test']['input_ids'][idx], skip_special_tokens=False)}")
lbl = tokenized_datasets["test"]["labels"][idx]
lbl = [v if v!=-100 else tokenizer.eos_token_id for v in lbl]
logger.info(f"labels: {tokenizer.decode(lbl, skip_special_tokens=True)}")
logger.info(f"labels ids: {tokenized_datasets['test']['labels'][idx]}")
if(use_key):
    key_lbl = tokenized_datasets['test']['key_ids'][idx]
    key_lbl = [v if v!=-100 else tokenizer.eos_token_id for v in key_lbl]
    logger.info(f"key_ids: {tokenizer.decode(key_lbl, skip_special_tokens=False)}")
    logger.info(f"key_ids: {tokenized_datasets['test']['key_ids'][idx]}")
logger.info("-"*30)

logger.info("-"*30)

tokenized_datasets.set_format(type="torch", columns=tokenized_datasets["train"].column_names)
logger.info(tokenized_datasets)
print("Tokenization Done")
logger.info("Tokenization Done")
logger.info("-"*30)

#---------------------------- 

## Model Definition
class Model(torch.nn.Module):
    def __init__(self, model_name, tokenizer, use_key, hidden_size, rep_token_id):
        super(Model, self).__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(model_name)
        self.lm.resize_token_embeddings(len(tokenizer))
        
        if(use_key):
            self.rep_token_id = rep_token_id
            self.bow_head = torch.nn.Linear(hidden_size, len(tokenizer), bias=False)
        
    def forward(self, input_ids, labels, key_ids = None):
        if(use_key):
            lm_out = self.lm(input_ids=input_ids, labels=labels, output_hidden_states=True)
            
            idx = (input_ids[0]==rep_token_id).nonzero()[0][0].item()
            hidden = lm_out.hidden_states[-1]
            h = hidden[0][idx]
            bow_logits = self.bow_head(h)
            
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            bow_ids = key_ids[0]
            bow_logits = bow_logits.expand(bow_ids.size(0), -1)
            
            bow_loss = loss_fct(bow_logits, bow_ids)
            loss = (bow_wt*bow_loss) + lm_out.loss
        else:
            lm_out = self.lm(input_ids=input_ids, labels=labels)
            loss = lm_out.loss
        return {'loss': loss}
    
def model_init():
    model = Model(MODEL_CKPT, tokenizer, use_key, hidden_size, rep_token_id)
    return model

#----------------------------

## Training
args = TrainingArguments(
    output_dir=model_dir,
    seed=10,
    label_names = lst_label_names,
    prediction_loss_only = True,
    evaluation_strategy="epoch",
    logging_dir = log_dir,
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    weight_decay=0.01,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=batch_size,
    save_total_limit=1,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    report_to=["tensorboard"]
)

trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=default_data_collator,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 2)]
)

print("Training...")
logger.info("Training...")
trainer.train()
logger.info("Training Done")

df_log = pd.DataFrame(trainer.state.log_history)
fl = os.path.join(model_dir, "log_history.csv")
df_log.to_csv(fl)  

best_model_ckpt = trainer.state.best_model_checkpoint
logger.info(f"Best model: {best_model_ckpt}")
tokenizer.save_pretrained(best_model_ckpt)

opt_file = os.path.join(best_model_ckpt, "optimizer.pt")
if os.path.exists(opt_file):
    os.remove(opt_file)
    logger.info(f"Removed {opt_file}")

print("Training done.")
#----------------------------