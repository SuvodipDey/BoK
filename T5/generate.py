from datasets import Dataset, DatasetDict, load_dataset
import transformers
from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
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
Generate response for test data:-

1. For models using BoK/BoW loss 
python generate.py -path=<model_dir> -key

2. For base model (without BoK/BoW loss) 
python generate.py -path=<model dir>
"""
#----------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-path','--path', help='path of the model directiory', required=True)
parser.add_argument('-dt','--dt', help='Name of dataset: dd (DailyDialg), pc (PersonaChat), emo (EmoWOZ)', required=True, choices=['dd', 'pc', 'emo'])
parser.add_argument('-model','--model', help='T5 model name', default='large', choices=['small', 'base', 'large'])
parser.add_argument('-best_model','--best_model', help='path of best model (already trained)', required=False, default="")
parser.add_argument('-key','--key', help='Use keywords', default=False, action='store_true')
parser.add_argument('-test','--test', help='Test Run', default=False, action='store_true')
parser.add_argument('-lbl','--lbl', help='Label for result directory', required=False, default="default")
parser.add_argument('-batch','--batch', help='Batch size', type=int, required=False, default=8)
parser.add_argument('-gmax','--gmax', help='Maximum length for generation', type=int, required=False, default=41)
parser.add_argument('-gmin','--gmin', help='Minimum length for generation', type=int, required=False, default=12)
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

new_tokens = {'eou':'<eou>'}
if(dataset=="pc"):
    new_tokens['knlg'] = '<knlg>'
if(use_key):
    new_tokens['key'] = '<key>'
    new_tokens['nok'] = '<nok>'
        
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
    def __init__(self, model_name, tokenizer, use_key, hidden_size):
        super(Model, self).__init__()
        self.lm = T5ForConditionalGeneration.from_pretrained(model_name)
        self.lm.resize_token_embeddings(len(tokenizer))
        
        if(use_key):
            self.bow_head = torch.nn.Linear(hidden_size, len(tokenizer), bias=False)
        
    def forward(self, input_ids, attention_mask, labels, key_ids = None):
        if(use_key):
            lm_out = self.lm(input_ids=input_ids, attention_mask=attention_mask, 
                           labels=labels, output_hidden_states=True)
            hidden = lm_out.decoder_hidden_states[-1]
            h = torch.permute(hidden, (1, 0, 2))[0]
            fc1 = self.bow_head(h)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            
            b_size = key_ids.size(0)
            bow_loss = 0
            for i in range(b_size):
                bow_ids = key_ids[i]
                bow_logits = fc1[i]
                bow_logits = bow_logits.expand(bow_ids.size(0), -1)
                b_loss = loss_fct(bow_logits, bow_ids)
                bow_loss+=b_loss
            bow_loss = bow_loss/b_size
            loss = (bow_wt*bow_loss) + lm_out.loss
        else:
            lm_out = self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = lm_out.loss
        return {'loss': loss}
    
#----------------------------

hidden_size = 1024
if(model_name=="small"):
    hidden_size = 512
elif(model_name=="base"):
    hidden_size = 768
    
MODEL_CKPT = f"t5-{model_name}"
tokenizer = AutoTokenizer.from_pretrained(best_model_path)
print("Tokenizer loaded.")

raw_model = Model(MODEL_CKPT, tokenizer, use_key, hidden_size)
raw_model.to(device)
m_path = os.path.join(best_model_path, "pytorch_model.bin")
raw_model.load_state_dict(torch.load(m_path))
raw_model.eval()
model = raw_model.lm
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
    df_input = df["input"]
    trm_input = trim_context(df_input, tokenizer.tokenize(df_input), max_len)
    output = tokenizer(trm_input, max_length=max_len, truncation=True)
    return output

#----------------------------

## Load Test Data
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

## Generate Response
logger.info("Generating Output...")
data_collator = DataCollatorForSeq2Seq(tokenizer)
data_loader = DataLoader(tokenized_datasets, batch_size, shuffle=False, collate_fn=data_collator)

predictions = []
for dt in tqdm(data_loader):
    input_ids = dt["input_ids"].to(device)
    attention_mask = dt["attention_mask"].to(device)
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, 
                            num_beams=beam, max_new_tokens=gmax, min_new_tokens=gmin, length_penalty=pen)

    out = tokenizer.batch_decode(output, skip_special_tokens=True)
    predictions += out
        
logger.info("Generation Complete.")
result_reply = os.path.join(result_dir, f"hyp_{result_label}.txt")

f = open(result_reply, "w") 
for line in predictions:
    f.write(line.strip()+"\n")
f.close()

logger.info("Generation done.")
print("Generation done.")
    
print("done")
#----------------------------