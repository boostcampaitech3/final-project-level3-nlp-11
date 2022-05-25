from model import VQAModel
from transformers import (AutoConfig,AutoModel,VisualBertForQuestionAnswering,AutoTokenizer,VisualBertConfig, TrainingArguments,get_linear_schedule_with_warmup)
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from torch.optim import AdamW
import torch.nn as nn
from tqdm import tqdm, trange
from tools import *
from dataset import VQADataset

import torch






def main():

    args = TrainingArguments(
        output_dir='models',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        
    )

    TOKENIZER_NAME='klue/bert-base'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_name='vqa_demo'
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    config  = VisualBertConfig()
    #config = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa")
    config.visual_embedding_dim = 2048
    config.num_labels = 2423
    config.vocab_size = tokenizer.vocab_size


    model = VQAModel(config)
    #model = VisualBertForQuestionAnswering(config)
    model.to(device)
    
    train_datasets,train_id2idx,train_idx2id = load_dataset('/opt/ml/Project/VQA/data/KVQA_annotations_train.json','train')
    val_datasets,val_id2idx,val_idx2id = load_dataset('/opt/ml/Project/VQA/data/KVQA_annotations_val.json','val')
    train_dataset = VQADataset(tokenizer,train_datasets,train_id2idx)
    val_dataset = VQADataset(tokenizer,val_datasets,val_id2idx)
                                                      

    #visual_embeds_dir = ['/opt/ml/Project/VQA/visualbert/featrues/KVQA_resnet101_faster_rcnn_genome-002.tsv',
    #                    '/opt/ml/Project/VQA/visualbert/featrues/VizWiz_resnet101_faster_rcnn_genome-002.tsv']

    trained_model = train(args,model,train_dataset,val_dataset)

    save_pth=f'/opt/ml/Project/VQA/models/{model_name}.pth'
    torch.save(trained_model.state_dict(),save_pth)





def train(args,model, datasets,val_datasets):

    train_sampler = RandomSampler(datasets)
    train_dataloader = DataLoader(datasets, sampler=train_sampler, batch_size=args.per_device_train_batch_size)

    val_sampler=RandomSampler(val_datasets)
    val_dataloader = DataLoader(val_datasets, sampler=val_sampler, batch_size=args.per_device_eval_batch_size)


    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)


    # Start training!
    global_step = 0
    
    model.zero_grad()
    torch.cuda.empty_cache()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        total_loss=0
        steps=0

        for _, batch in enumerate(epoch_iterator):
        

            steps+=1
            model.train()
            images= batch[3]
            if torch.cuda.is_available():
                batch = tuple(t.cuda()  for t in batch if type(t) != tuple)
            inputs = {'input_ids': batch[0],
                        'token_type_ids': batch[1],
                        'attention_mask': batch[2],
                        'image' : images
                        }
            

            target = batch[-1]
            outputs = model(**inputs,labels=target)

            
            loss = outputs.loss
            total_loss+=loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            torch.cuda.empty_cache()

        torch.save(model.state_dict(), f'/opt/ml/Project/VQA/models/vqa_demo_epoch{epoch+1}.pth')
        print(f'train loss : {total_loss/steps}')

        with torch.no_grad():
            print('Calculating Valdiation Result............')
            model.eval()
            with open('/opt/ml/Project/VQA/data/trainval_label2ans.kvqa.pkl','rb') as f :
                label2ans = pickle.load(f)

            val_loss=0
            val_step=0
            exact_match=0
            for _, batch in enumerate(tqdm(val_dataloader)):
                val_step+=1
                images= batch[3]
                if torch.cuda.is_available():
                    batch = tuple(t.cuda()  for t in batch if type(t) != tuple)
                inputs = {'input_ids': batch[0],
                            'token_type_ids': batch[1],
                            'attention_mask': batch[2],
                            'image' : images
                            }
                target = batch[-1]
                outputs = model(**inputs,labels=target)
                loss = outputs.loss
                logits=outputs.logits
                pred_idx = logits.argmax(-1)
                val_loss+=loss
                
                for step, idx in enumerate(pred_idx):
                    if target[step][idx] != 0:
                        exact_match+=1


            val_loss /= val_step
            exact_match /= len(val_datasets)
            print(f'Validation loss : {val_loss}')
            print(f'Validation Exact match : {exact_match*100}%')
        

        
    return model
    


if __name__ == '__main__':
    main()
