# BoK: Introducing Bag-of-Keywords Loss for Interpretable Dialogue Response Generation
Official code repository of the paper titled [BoK: Introducing Bag-of-Keywords Loss for Interpretable Dialogue Response Generation](https://aclanthology.org/2024.sigdial-1.48/).

## Install dependencies
Python 3.11 or later.
```console
❱❱❱ pip install -r requirements.txt
```

## Download Datasets
Download the following datasets. 

1. DailyDialog: Download link http://yanran.li/files/ijcnlp_dailydialog.zip
2. PersonaChat: Download data using ParlAI (https://parl.ai/docs/tasks.html#persona-chat)

Set the dataset paths correctly in the following files: DialoGPT/create_data.py and T5/create_data.py

## Train
Train GPT2 and T5 by running train.py script provided in the respective directories.

1. With BoK loss
```console
❱❱❱ python train.py -path=<model_dir> -src_file=train.py -dt=dd/pc -key
```

2. With BoW loss
```console
❱❱❱ python train.py -path=<model_dir> -src_file=train.py -dt=dd/pc -key -all
```

3. Basic Model
```console
❱❱❱ python train.py -path=<model_dir> -src_file=train.py -dt=dd/pc
```
