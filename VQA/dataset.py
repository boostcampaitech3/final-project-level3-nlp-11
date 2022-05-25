from torch.utils.data import Dataset
from tools import *

import torch


class VQADataset(Dataset):
    def __init__(self,tokenizer,datasets,id2idx):
        super(VQADataset, self).__init__()
        viz_path = '/opt/ml/Project/VQA/data/VizWiz/'
        kvqa_path = '/opt/ml/Project/VQA/data/kvqa/images_v1.0 (1)/kvqa_resize_more/'
        
        self.images=[ viz_path+file if file[:6]== 'VizWiz' else kvqa_path+file for file in list(datasets['image'])]
        self.inputs= tokenize(tokenizer,datasets)
        self.labels=list(datasets['labels'])
        self.scores=list(datasets['scores'])
        self.num_ans_candidates = 2423
    



    def __getitem__(self,index):
        q_input=self.inputs['input_ids'][index]
        q_token=self.inputs['token_type_ids'][index]
        q_attention=self.inputs['attention_mask'][index]
        label=torch.tensor(self.labels[index])
        score=torch.Tensor(self.scores[index])
        image= self.images[index]

        if len(label) !=0 :
            target = torch.zeros(self.num_ans_candidates)
            target.scatter_(0, label, score)
            
            return q_input,q_token,q_attention,image, target
        else:
            target = torch.zeros(self.num_ans_candidates)
            label = torch.tensor([1])
            score = torch.Tensor([1])
            target.scatter_(0, label, score)
            return q_input,q_token,q_attention,image, target




    def __len__(self):
        return len(self.images)
