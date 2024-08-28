import os
from datasets import Dataset, DatasetDict, load_dataset
import re
import spacy
from functools import partial
from nltk.corpus import stopwords
import collections

#----------------------------

test_count = 50

### DailyDialog Path ###
daily_dialog_path = "../ijcnlp_dailydialog"
PERSONACHAT_PATH = "../personachat"
max_words = 100
delim = " "

lst_pos_tags = ['NN', 'NNP', 'NNS', 'JJ', 'CD', 'VB', 'VBN', 'VBD', 'VBG', 'RB', 'VBP', 'VBZ', 'NNPS', 'JJS']
stop_words = stopwords.words('english')
nlp = spacy.load("en_core_web_sm")

#----------------------------

def tokenize_sentence(txt, tokenizer):
    result = tokenizer(txt)
    word_ids = result.word_ids()
    if tokenizer.is_fast:
        result["word_ids"] = [word_ids[i] for i in range(len(result["input_ids"]))]
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

def get_pos_tags(doc):
    pos_tags = {}
    for token in doc:
        if(not (token.is_stop or token.is_punct or token.is_space or token.text.lower() in stop_words)):
            if(token.tag_ in lst_pos_tags):
                pos_tags[token.text.lower()] = token.tag_
    return pos_tags
                    
def prep_out(df, special_tokens, all_key, custom_kw_extractor, add_pos, tokenizer):
    response = df["output"]
    output = {"key_words": []}
    
    for i in range(len(response)):
        txt = response[i].strip()
        if(all_key):
            output["key_words"].append(txt.lower())
        else:
            keywords = custom_kw_extractor.extract_keywords(txt)
            str_kw = ""
            kw_list = []
            for kw,_ in keywords:
                kw = kw.lower()
                arr = kw.split(" ")
                if(len(arr)<3):
                    if kw not in str_kw:
                        str_kw = str_kw + delim + kw
                        kw_list.append(kw)
                else:
                    str_kw = str_kw + delim + kw
                    kw_list.append(kw)
            
            if(add_pos):
                txt_doc = nlp(txt)
                pos_tags = get_pos_tags(txt_doc)
                for w in pos_tags:
                    if(w not in kw_list):
                        kw_list.append(w)
            
            if(len(kw_list)==0):
                key_words = special_tokens['nok']
            else:
                key_words = delim.join(kw_list)
            output["key_words"].append(key_words.lower().strip())
    
    return output

def format_utterance(utt):
    utt_text = utt.replace("’","'").strip()
    utt_text = re.sub("(\s*)([a-zA-Z|0-9]+)(\.| \.|\. )([a-zA-Z]+)(\s*)", r"\1\2 . \4\5", utt_text)
    arr = utt_text.split()
    return " ".join(arr[0:max_words]).strip()

#----------------------------

## DailyDialog
def load_dd_data(mode, special_tokens, test_run):
    data = []
    path = os.path.join(daily_dialog_path, mode, f"dialogues_{mode}.txt")
    path_act = os.path.join(daily_dialog_path, mode, f"dialogues_act_{mode}.txt")
    path_emo = os.path.join(daily_dialog_path, mode, f"dialogues_emotion_{mode}.txt")
    
    with open(path, "r") as f:
        for line in f:
            ut_lst = line.replace("\n", " ").split(" __eou__ ")[:-1]
            utt_list = [format_utterance(utt) for utt in ut_lst]
            data.append(utt_list)
            
    data_dict = {"input": [], "output": []}
    c = 0
    n = 0            
    for conv in data:
        c+=1
        for i in range(len(conv)-1):
            context = special_tokens['eou'].join(conv[:i+1]) + special_tokens['eou']
            data_dict["input"].append(context.strip())    
            data_dict["output"].append(conv[i+1].strip())
            n+=1
        if(test_run and c==test_count):
            break
    
    return data_dict

#----------------------------

## PersonaChat
def load_personachat(mode, special_tokens, test_run):
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
    data_dict = {"input": [], "output": []}
    for k in persona_data:
        conv = persona_data[k]['dialog']
        p_spk0 = " ".join(persona_data[k]['p_partner_spk0'])
        p_spk1 = " ".join(persona_data[k]['p_self_spk1'])
        
        c+=1
        for i in range(len(conv)-1):
            context = special_tokens['eou'].join(conv[:i+1]) + special_tokens['eou']
            persona = p_spk0 if (i%2==1) else p_spk1
            context = context.strip() + special_tokens['knlg'] + persona.strip()
            
            data_dict["input"].append(context)
            data_dict["output"].append(conv[i+1].strip())
            n+=1
        if(test_run and c==test_count):
            break
    return data_dict

#----------------------------

def get_data(dataset, special_tokens, use_key, all_key, custom_kw_extractor, add_pos, tokenizer, test_run):
    if(dataset=="pc"):
        train_dataset = Dataset.from_dict(load_personachat("train", special_tokens, test_run))
        valid_dataset = Dataset.from_dict(load_personachat("valid", special_tokens, test_run))
        test_dataset = Dataset.from_dict(load_personachat("test", special_tokens, test_run))
    else:
        train_dataset = Dataset.from_dict(load_dd_data("train", special_tokens, test_run))
        valid_dataset = Dataset.from_dict(load_dd_data("validation", special_tokens, test_run))
        test_dataset = Dataset.from_dict(load_dd_data("test", special_tokens, test_run))
    
    dataset = DatasetDict()
    dataset["train"] = train_dataset
    dataset["validation"] = valid_dataset
    dataset["test"] = test_dataset
    
    if(use_key):
        dataset = dataset.map(partial(prep_out, special_tokens=special_tokens, 
                                        all_key=all_key, custom_kw_extractor=custom_kw_extractor, 
                                        add_pos=add_pos, tokenizer=tokenizer), batched=True)
    return dataset
    
#----------------------------