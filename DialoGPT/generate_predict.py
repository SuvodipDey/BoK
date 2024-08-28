from datasets import Dataset, DatasetDict, load_dataset
import transformers
from transformers import AutoTokenizer, GPT2LMHeadModel, default_data_collator
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

#----------------------------

"""
Generates response along with top-k predicted words or keywords:-

python generate_predict.py -path=<model_dir> -key
"""

#----------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-path','--path', help='path of the model directiory', required=True)
parser.add_argument('-dt','--dt', help='Name of dataset: dd (DailyDialg) or pc (PersonaChat)', required=True, choices=['dd', 'pc'])
parser.add_argument('-model','--model', help='T5 model name', default='large', choices=['small', 'medium', 'large'])
parser.add_argument('-best_model','--best_model', help='path of best model (already trained)', required=False, default="")
parser.add_argument('-key','--key', help='Use keywords', default=False, action='store_true')
parser.add_argument('-test','--test', help='Test Run', default=False, action='store_true')
parser.add_argument('-lbl','--lbl', help='Label for result directory', required=False, default="default")
parser.add_argument('-batch','--batch', help='Batch size', type=int, required=False, default=8)
parser.add_argument('-gmax','--gmax', help='Maximum length for generation', type=int, required=False, default=40)
parser.add_argument('-gmin','--gmin', help='Minimum length for generation', type=int, required=False, default=11)
parser.add_argument('-beam','--beam', help='Beam size', type=int, required=False, default=5)
parser.add_argument('-pen','--pen', help='Length penalty', type=float, required=False, default=0.1)
parser.add_argument('-topn','--topn', help='top-n keywords', type=int, required=False, default=8)

args = vars(parser.parse_args())
model_dir = args['path']
dataset = args['dt']
model_name = args['model']
best_model_path = args['best_model']
use_key = args['key']
test_run = args['test']
result_label = args['lbl']
batch_size = args['batch']
gmax = args['gmax']
gmin = args['gmin']
beam = args['beam']
pen = args['pen']
topn = args['topn']

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print(f"Using GPU!")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

max_len = 512
max_len_response = 64
        
if(os.path.isdir(model_dir)):
    print(f"Model Directory {model_dir} exists.")
else:
    print(f"Model directory {model_dir} does not exists.")
    exit(0)
    
if(os.path.isdir(best_model_path)):
    print(f"Path of best model exists: {best_model_path}")
else:
    print(f"Path of best model does not exists : {best_model_path}")
    exit(0)

result_dir = os.path.join(model_dir, "result_" + result_label)
if(os.path.isdir(result_dir)):
    print(f"Result Directory {result_dir} exists.")
    #exit(0)
else:
    os.mkdir(result_dir)
    print(f"Result directory {result_dir} created.")

#----------------------------

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
    
#----------------------------

MODEL_CKPT = "microsoft/DialoGPT-small"    
hidden_size = 768   
if(model_name=="medium"):
    MODEL_CKPT = "microsoft/DialoGPT-medium"
    hidden_size = 1024
elif(model_name=="large"):
    MODEL_CKPT = "microsoft/DialoGPT-large"
    hidden_size = 1280
    
tokenizer = AutoTokenizer.from_pretrained(best_model_path)
print("Tokenizer loaded.")

new_tokens = {'eou': tokenizer.eos_token, 'rep': '<rep>'}
if(dataset=="pc"):
    new_tokens['knlg'] = tokenizer.eos_token
if(use_key):
    new_tokens['nok'] = '<nok>'
    
rep_token_id = tokenizer.convert_tokens_to_ids(new_tokens['rep'])
model = Model(MODEL_CKPT, tokenizer, use_key, hidden_size, rep_token_id)
model.to(device)
m_path = os.path.join(best_model_path, "pytorch_model.bin")
model.load_state_dict(torch.load(m_path))
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {n}")
model.eval()
print("Model loaded.")

#----------------------------

#Setting log file
log_file = os.path.join(result_dir, 'log.txt')
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
logger.info(f"best_model_path: {best_model_path}")
logger.info("Model and Tokenizer Loaded")
print("Model and Tokenizer Loaded")
logger.info("-"*30)
                    
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
    trm_input = trim_context(df_input, tokenizer.tokenize(df_input), max_len)
    input_ids = tokenizer.encode(trm_input + new_tokens['rep'])
    output = {}
    output["input_ids"] = input_ids
    output["attention_mask"] = [1]*len(input_ids)
    return output

#----------------------------

## Load Test data

print("Tokenizing Data ...")
logger.info("Tokenizing Data ...")
df = get_data(dataset, new_tokens, False, False, None, False, None, test_run)

logger.info(df)
logger.info("-"*30)
logger.info("Data Snippet:-")
idx=4
logger.info(df["test"]["input"][idx])
logger.info(df["test"]["output"][idx])
logger.info("-"*30)
logger.info(df["test"]["input"][idx+1])
logger.info(df["test"]["output"][idx+1])
logger.info("-"*30)
logger.info("-"*30)

test_data = df["test"]
tokenized_datasets = test_data.map(prepare_features, remove_columns=test_data.column_names)
tokenized_datasets.set_format(type="torch")
logger.info(tokenized_datasets)
print("Tokenization Done")
logger.info("Tokenization Done")
logger.info("-"*30)

#----------------------------

## Generate Response and predict top-k keywords

logger.info("Generating Output...")
data_loader = DataLoader(tokenized_datasets, 1, shuffle=False, collate_fn=default_data_collator)

predictions = []
lst_key_tokens = []
lst_prob = []

for dt in tqdm(data_loader):
    input_ids = dt["input_ids"].to(device)
    attention_mask = dt["attention_mask"].to(device)
    
    with torch.no_grad():
        output = model.lm.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam, 
                                max_new_tokens=gmax, min_new_tokens=gmin, length_penalty=pen)
            
        lm_out = model.lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        hidden = lm_out.hidden_states[-1]
        h = hidden[0][-1]
        fc1 = model.bow_head(h)
        score = torch.nn.functional.softmax(fc1)

        val, indices = torch.topk(score, 8)
        lst_idx = indices.tolist()
        lst_val = val.tolist()
        
    out = tokenizer.batch_decode(output, skip_special_tokens=False)
    predictions += out
    lst_key_tokens.append(lst_idx)
    lst_prob.append(lst_val)
    
logger.info("Generation Complete.")
result_reply = os.path.join(result_dir, f"hyp_{result_label}.txt")

n = len(predictions)
f = open(result_reply, "w") 
for i in range(n):
    resp = predictions[i].split(new_tokens['rep'])[1]
    resp = resp.replace(tokenizer.eos_token,'')
    key = tokenizer.convert_ids_to_tokens(lst_key_tokens[i])
    
    lst_idx_val = []
    for j in range(len(key)):
        lst_idx_val.append((key[j], round(lst_prob[i][j],4))) 
    s_out = f"{resp.strip()} <keywords> {lst_idx_val}\n"
    f.write(s_out)
f.close()

logger.info("Generation done.")
print("Generation done.")
    
print("done")
#----------------------------