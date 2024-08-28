import os
import pandas
import math
import random
import torch
import time
from transformers import AutoTokenizer, RobertaForMaskedLM
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler
import argparse
import re
import collections
from scipy.stats import pearsonr, spearmanr
#from prepare_eval_data import load_tc_usr, load_pc_usr, load_engage_data, load_holistic_data
import torch.nn.functional as F
import spacy
from nltk.corpus import stopwords
#import shutil
import yake
from nltk.tokenize import word_tokenize
from tqdm import tqdm

#----------------------------

"""
Dial-M Evaluation:-
Download the Dial-M model before running the evaluation. Follow https://github.com/SuvodipDey/Dial-M.

python eval_dialm.py -path=<output_dialm> -dt=dd/pc -out=<out_dir> -out=<generated_resonse_file> -lbl=<hypothesis_file>

Pass the path of the trained model containing the output of the generate.py file in the -out argument. 
Pass dd_res_file.txt/pc_res_file.txt in the -lbl argument based on the dataset.

"""

#----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-path','--path', help='path of the model directiory', required=True)
parser.add_argument('-dt','--dt', help='Name of dataset: dd (DailyDialg), pc (PersonaChat), emo (EmoWOZ)', required=True, choices=['dd', 'pc', 'emo'])
parser.add_argument('-out','--out', help='name of output directory', required=True)
parser.add_argument('-lbl','--lbl', help='result label', required=True)
parser.add_argument('-keys','--keys', help='Number of keywords', type=int, required=False, default=20)
parser.add_argument('-no_pos','--no_pos', help='Skip POS keywords', default=True, action='store_false')
parser.add_argument('-no_pre','--no_pre', help='Inference for model with no pre-training', default=False, action='store_true')
args = vars(parser.parse_args())
model_dir = args['path']
dataset = args['dt']
out_dir = args['out']
numOfKeywords = args['keys']
add_pos = args['no_pos']
out_label = args['lbl']
no_pre = args['no_pre']
res_dir = os.path.join(out_dir, f"result_{out_label}")
res_file = os.path.join(res_dir, f"hyp_{out_label}.txt")
test_run=True

if(model_dir=="roberta-base"):
    print("Evaluating with roberta-base model.")
    model_dir = "roberta-base"
    #out_dir = "result_roberta_base"
    no_pre = True
else:
    if(not os.path.isdir(model_dir)):
        print("Model Directory does not exist.")
        exit(0)
        
if(not os.path.isdir(res_dir)):
    print("Output Directory does not exist.")
    exit(0)

if(not os.path.isfile(res_file)):
    print("Result file does not exist.")
    exit(0)

print(f"Model directory: {model_dir}")
print(f"Output directory: {res_dir}")
print(f"Result file: {res_file}")
print(f"numOfKeywords: {numOfKeywords}")
print(f"no_pre: {no_pre}")
    
#----------------------------

eou_token = "<eou>"
knlg_token = "<knlg>"
max_len=512

lst_pos_tags = ['NN', 'NNP', 'NNS', 'JJ', 'CD', 'VB', 'VBN', 'VBD', 'VBG', 'RB', 'VBP', 'VBZ', 'NNPS', 'JJS']
stop_words = stopwords.words('english')

SEED = 10
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)      
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
if(no_pre):
    eou_token = tokenizer.eos_token
model = RobertaForMaskedLM.from_pretrained(model_dir)
model.to(device)
model.eval()

language = "en"
max_ngram_size = 1
deduplication_threshold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
nlp = spacy.load("en_core_web_sm")
print("Model Loaded")

#----------------------------

def tokenize_sentence(txt, tokenizer):
    result = tokenizer(txt)
    word_ids = result.word_ids()
    if tokenizer.is_fast:
        result["word_ids"] = [word_ids[i] for i in range(len(result["input_ids"]))]
        for i in range(len(result["input_ids"])):
            if(result["input_ids"][i] >= 50265):
                result["word_ids"][i] = None
                break
    return result

def tokenize_sentence_truncated(txt, tokenizer, n):
    result = tokenizer(txt, truncation=True, max_length=n)
    word_ids = result.word_ids()
    if tokenizer.is_fast:
        result["word_ids"] = [word_ids[i] for i in range(len(result["input_ids"]))]
        for i in range(len(result["input_ids"])):
            if(result["input_ids"][i] >= 50265):
                result["word_ids"][i] = None
                break
    return result

def get_word_mapping(tok):
    word_ids = tok["word_ids"].copy()
    mapping = collections.defaultdict(list)
    current_word_index = -1
    current_word = None
    for idx, word_id in enumerate(word_ids):
        if word_id is not None:
            if word_id != current_word:
                current_word = word_id
                current_word_index += 1
            mapping[current_word_index].append(idx)
    return mapping

def get_pos_tags(doc, m_type):
    pos_tags = {}
    for token in doc:
        if(m_type==1):
            if(not (token.is_stop or token.is_punct or token.is_space or token.text.lower() in stop_words)):
                if(token.tag_ in lst_pos_tags):
                    pos_tags[token.text.lower()] = token.tag_
        else:
            if(not (token.is_punct or token.is_space)):
                pos_tags[token.text.lower()] = token.tag_
    return pos_tags

def get_mask_words(txt, tok, mapping, add_pos):
    yake_doc = txt.replace(eou_token, " ")
    yake_doc = yake_doc.strip()
    keywords = custom_kw_extractor.extract_keywords(yake_doc)
    lst_kw = [kw[0] for kw in keywords]
    
    txt_doc = nlp(txt)
    if(len(lst_kw)<numOfKeywords and add_pos):
        n = numOfKeywords-len(lst_kw)
        pos_tags = get_pos_tags(txt_doc, 1)
        for w in pos_tags:
            if(w not in lst_kw):
                lst_kw.append(w)
                n = n-1
                if(n==0):
                    break
    
    mask = []
    mask_words = []
    for idx in mapping:
        start, end = tok.word_to_chars(idx)
        word = txt[start:end].lower()
        if word in lst_kw:
            mask.append(idx)
            mask_words.append(word)
            
    if(len(mask)==0):
        lst_kw = []
        n = numOfKeywords
        pos_tags = get_pos_tags(txt_doc, 2)
        for w in pos_tags:
            lst_kw.append(w)
            n = n-1
            if(n==0):
                break
                
        for idx in mapping:
            start, end = tok.word_to_chars(idx)
            word = txt[start:end].lower()
            if word in lst_kw:
                mask.append(idx)
                mask_words.append(word)
                
        if(len(mask)==0):
            for idx in mapping:
                start, end = tok.word_to_chars(idx)
                word = txt[start:end].lower()
                mask.append(idx)
                mask_words.append(word)
            
    return mask, mask_words
    
def get_masked_tokens(tokenizer, tok, mapping, mask):
    mask_ids = []
    input_ids = tok["input_ids"].copy()
    labels = [-100]*len(input_ids)
    for word_id in mask:
        for idx in mapping[word_id]:
            mask_ids.append(input_ids[idx])
            labels[idx] = input_ids[idx]
            input_ids[idx] = tokenizer.mask_token_id
    return input_ids, labels

def evaluate(input_id, lbl, attn_mask):
    input_ids = torch.tensor([input_id], dtype=torch.long).to(device)
    labels = torch.tensor([lbl], dtype=torch.long).to(device)
    attention_masks = torch.tensor([attn_mask], dtype=torch.long).to(device)
    loss = 0.0
    with torch.no_grad():
        output = model(input_ids = input_ids, attention_mask = attention_masks, labels = labels)
        loss = output.loss.item()
    return loss

def get_score(prev, resp, tok_context, tok_prev, tok_resp, tok_condition, use_condition, logger):
    map_resp = get_word_mapping(tok_resp)
    mask, mask_words = get_mask_words(resp, tok_resp, map_resp, add_pos)
    
    score = -1
    if(len(mask)>0):
        total_score = 0
        for word_id in mask:
            resp_masked, lbl_resp = get_masked_tokens(tokenizer, tok_resp, map_resp, [word_id])
            tok1 = []
            j=0
            if(len(tok_context)>0):
                tok1.extend(tok_context.copy()[j:-1])
                j=1
            if(prev is not None):
                tok1.extend(tok_prev["input_ids"].copy()[j:-1])
                j=1
            lbl1 = [-100]*len(tok1)
            tok1.extend(resp_masked[j:])
            lbl1.extend(lbl_resp[j:])

            if(use_condition):
                tok_kn = tok_condition["input_ids"].copy()
                tok_kn[0] = tokenizer.sep_token_id
                tok1.extend(tok_kn)
                lbl1.extend([-100]*len(tok_kn))
            attn1 = [1]*len(tok1)
            score = evaluate(tok1, lbl1, attn1)
            total_score+= score
        score = total_score/len(mask)
    return round(score, 4)

def get_metric(txt_input, response, logger):
    utt_list = []
    context = None
    condition = None
    use_condition = False
    
    if(knlg_token in txt_input):
        arr  = txt_input.split(knlg_token)
        context  = arr[0].strip()
        condition = arr[1].strip()
        use_condition = True
    else:
        context = txt_input
        
    arr = context.split(eou_token)
    utt_list = []
    for i in range(len(arr)-1):
        utt = arr[i].strip()
        utt_list.append(utt)
    
    if(response is not None):
        resp = response
    else:
        resp = utt_list[-1]
    prev = None
    if(len(utt_list)>1):
        prev = utt_list[-2]
    
    resp = f"{resp}{eou_token}"
    tok_resp = tokenize_sentence(resp, tokenizer)
    
    tok_prev = []
    if(prev is not None):
        prev = f"{prev}{eou_token}"
        tok_prev = tokenize_sentence(prev, tokenizer)
    
    tok_condition = None
    tok_count = 0
    if(use_condition):
        tok_condition = tokenize_sentence(condition, tokenizer)
        if(prev is not None):
            tok_count = len(tok_prev["input_ids"]) + len(tok_resp["input_ids"]) + len(tok_condition["input_ids"]) - 2
        else:
            tok_count = len(tok_resp["input_ids"]) + len(tok_condition["input_ids"]) - 2
    else:
        if(prev is not None):
            tok_count = len(tok_prev["input_ids"]) + len(tok_resp["input_ids"]) - 2
        else:
            tok_count = len(tok_resp["input_ids"]) - 2
        
    if(tok_count>max_len):
        if(use_condition):
            if(prev is not None):
                n = max_len - len(tok_resp["input_ids"]) - len(tok_resp["input_ids"])
            else:
                n = max_len - len(tok_resp["input_ids"])
            tok_condition = tokenize_sentence_truncated(condition, tokenizer, n)
        else:
            print(f"Input length exceeded!!! {tok_count}")
            logger.write("Input length exceeded!!!\n")
            logger.write(f"tok_count: {tok_count}\n")
            h = len(tok_resp["input_ids"])
            logger.write(f"tok_resp: {h}\n")
            h = len(tok_prev["input_ids"])
            logger.write(f"tok_prev: {h}\n")
            return -100
    
    tok_context = []
    context = ""
    if(len(utt_list)>2):
        con_list = []
        n = 0
        for k in range(len(utt_list)-3,-1,-1):
            utt_text = f"{utt_list[k]}{eou_token}"
            tok_utt = tokenizer(utt_text)
            if(n+len(tok_utt["input_ids"])+tok_count-2<=max_len):
                n += len(tok_utt["input_ids"])-2
                con_list.append(utt_text)
            else:
                break
        con_list.reverse()            
        context = "".join(con_list)
        tok_context = tokenizer(context)["input_ids"]
    
    logger.write(f"prev: {prev}\n")
    logger.write(f"resp: {resp}\n")
    if(use_condition):
        logger.write(f"condition: {condition}\n")
    
    score = get_score(prev, resp, tok_context, tok_prev, tok_resp, tok_condition, use_condition, logger)
    logger.write(f"Dial-M score: {score}\n")
    return score

def compute_correlation(lst_gt, lst_score, logger):
    corr1, p_val1 = pearsonr(lst_gt, lst_score)
    corr2, p_val2 = spearmanr(lst_gt, lst_score)
    logger.write(f"Correlation between GT and Dial-M score: Pearson = ({round(corr1,4)}, {round(p_val1,4)}), Spearman = ({round(corr2,4)}, {round(p_val2,4)})\n")
    
    logger.write("-"*30+"\n")
    logger.write("-"*30+"\n")
        
#----------------------------

def format_utterance(txt):
    txt = txt.replace("\n", "")
    pred_tokens = word_tokenize(txt.strip().lower())
    s = ' '.join(pred_tokens)
    s = s.replace(" s ", "s ")
    s = s.replace(" nt ", "nt ")
    s = s.replace(" m ", "m ")
    s = s.replace(" - ", "-")
    s = s.replace(" / ", "/")
    s = s.replace("`", "'")
    s = re.sub("(\s*)([a-zA-Z|0-9]+)(\.| \.|\. )([a-zA-Z]+)(\s*)", r"\1\2 . \4\5", s)
    return s

def evaluate_dd_data():
    data = []
    mode = "test"
    daily_dialog_path = "../DialM/raw_data/ijcnlp_dailydialog"
    data_path = os.path.join(daily_dialog_path, mode, f"dialogues_{mode}.txt")
    
    out_path = os.path.join(res_dir, f'dialm_{out_label}.txt')
    logger = open(out_path, "w")
    
    with open(data_path, "r") as f:
        for line in f:
            ut_lst = line.replace("\n", " ").split(" __eou__ ")[:-1]
            utt_list = [format_utterance(utt) for utt in ut_lst]
            data.append(utt_list)
    
    res = []
    with open(res_file, "r") as f:
        for line in f:
            utt = line.replace("\n", "")
            utt = format_utterance(utt.strip())
            res.append(utt)
    
    c = 0
    n = 0   
    nk = 0
    f_pr = []
    for conv in tqdm(data):        
        for i in range(len(conv)-1):
            context = eou_token.join(conv[:i+1])
            response = res[n]
            ctx = context.strip() + eou_token + response.strip() + eou_token
            score = get_metric(ctx, None, logger)
            if(score<0):
                nk+=1
                logger.write(f"No keywords !!! \n")
            else:
                f_pr.append(score)
            logger.write("-"*30+"\n")
            n+=1
            
        c+=1 
        #if(c==4):
        #    break
    
    logger.write("="*30+"\n")
    f_score = round(sum(f_pr)/len(f_pr), 4)
    logger.write(f"Final score: {f_score}\n")
    logger.write(f"no keywords: {nk}\n")
    logger.write(f"{len(res)}: {n}\n")
    logger.close()

#----------------------------

def evaluate_pc_data():
    out_path = os.path.join(res_dir, f'dialm_{out_label}.txt')
    logger = open(out_path, "w")
    
    res = []
    with open(res_file, "r") as f:
        for line in f:
            utt = line.replace("\n", "")
            utt = format_utterance(utt.strip())
            res.append(utt)
    
    PERSONACHAT_PATH = "../DialM/raw_data/personachat"
    mode = "test"
    file_name = os.path.join(PERSONACHAT_PATH, f"{mode}_both_original.txt")
    with open(file_name, 'r') as f:
        lines = f.readlines()
    persona_data = {}
    c = -1
    for line in lines:
        if("1 your persona:" in line):
            c+=1
            persona_data[c] = {}
            persona_data[c]['p_self_spk1'] = []
            persona_data[c]['p_partner_spk0'] = []
            persona_data[c]['dialog'] = []
        if(" persona:" in line):
            p = line.split(" persona:")[1]
            if(" your persona:" in line):
                persona_data[c]['p_self_spk1'].append(p.strip())
            else:
                persona_data[c]['p_partner_spk0'].append(p.strip())
        else:
            p = line.split("\t")
            c1 = p[0].strip()
            c1 = re.sub("^(\d)+ ", "", c1)
            c2 = p[1].strip()
            persona_data[c]['dialog'].append(format_utterance(c1))
            persona_data[c]['dialog'].append(format_utterance(c2))
    
    c = 0
    n = 0
    nk = 0
    f_pr = []
    #data_dict = {"input": [], "response": []}
    for k in tqdm(persona_data):
        conv = persona_data[k]['dialog']
        p_spk0 = " ".join(persona_data[k]['p_partner_spk0'])
        p_spk1 = " ".join(persona_data[k]['p_self_spk1'])
        
        c+=1
        context = ""
        for i in range(len(conv)-1):
            context = eou_token.join(conv[:i+1])
            gt = conv[i+1]
            response = res[n]
            persona = p_spk0 if (i%2==1) else p_spk1
            ctx = context.strip()+eou_token+response.strip()+eou_token+knlg_token+persona.lower().strip()
            score = get_metric(ctx, None, logger)
            logger.write(f"gt: {gt}\n")
            if(score<0):
                nk+=1
                logger.write(f"No keywords !!! \n")
            else:
                f_pr.append(score)
            logger.write("-"*30+"\n")
            n+=1
            
        c+=1 
        #if(c==4):
        #    break
            
    logger.write("="*30+"\n")
    f_score = round(sum(f_pr)/len(f_pr), 4)
    logger.write(f"Final score: {f_score}\n")
    logger.write(f"no keywords: {nk}\n")
    logger.write(f"{len(res)}: {n}\n")
    logger.close()       

#----------------------------

if(dataset == "dd"):
    evaluate_dd_data()
    
if(dataset == "pc"):
    evaluate_pc_data()

print("done")

#----------------------------