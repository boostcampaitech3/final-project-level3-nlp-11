from tqdm import tqdm

import feature_extract.utils
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import re
import torch
import numpy as np
import pandas as pd
import json
import pickle




def tokenize(tokenizer,datasets):
    tokenized_question = tokenizer(list(datasets['question']),
                                    return_tensors="pt",
                                    padding="max_length",
                                    truncation=True,
                                    max_length=256,
                                    add_special_tokens=True,              
                                    )
    return tokenized_question


def get_dataset_with_label(dataframe,type):
    with open(f'/opt/ml/Project/VQA/data/{type}_target.kvqa.pkl',"rb") as fr:
        target = pickle.load(fr)
    labels=[]
    scores=[]
    for t in target:
        label =t['labels']
        score = t['scores']
        labels.append(label)
        scores.append(score)
    dataframe['labels'] = labels
    dataframe['scores'] = scores
    #dataframe = remove_empty_label(dataframe)
    return dataframe

def remove_empty_label(dataframe):
    empty_labels=[]
    for idx, label in enumerate(dataframe['labels']):
        if len(label)==0:
            empty_labels.append(idx)
    dataframe = dataframe.drop(labels = empty_labels,axis=0)
    dataframe = dataframe.reset_index(drop=True)
    return dataframe

def load_dataset(datadir,type):
    with open(datadir, 'r') as f:
        json_data = json.load(f)
    df= pd.DataFrame(json_data)
    if type == 'test':
        pass
    else:
        df = get_dataset_with_label(df,type)

    id2idx={}
    idx2id={}
    for idx,image_id in enumerate(df['image']):
        id2idx[image_id]=idx
        idx2id[idx]=image_id


    return df,id2idx,idx2id
