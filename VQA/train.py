from load_data import *
from transformers import (AutoConfig,AutoModel,AutoTokenizer,TrainingArguments,get_linear_schedule_with_warmup)
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from tqdm import tqdm, trange


import torch






def main():

    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        num_train_epochs=12,
        weight_decay=0.01
    )

    Q_MODEL_NAME='klue/bert-base'
    model_name='vqa_model'

    config  = AutoConfig(Q_MODEL_NAME)
    Q_model = AutoModel.from_pretrained(Q_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(Q_MODEL_NAME)


    datasets = load_dataset('/opt/ml/data/KVQA_annotations_train.json')
    #img_datasets,question_datasets = load_dataset('/opt/ml/data/KVQA_annotations_train.json')

    trained_model = train(Q_model,datasets)

    save_pth=f'/opt/ml/Project/VQA/models/{model_name}.pth'
    torch.save(trained_model.load_state_dict(),save_pth)

    pass




def train(args,model, datasets):

    train_sampler = RandomSampler(datasets)
    train_dataloader = DataLoader(datasets, sampler=train_sampler, batch_size=args.per_device_train_batch_size)


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

        for step, batch in enumerate(epoch_iterator):
            steps+=1
            model.train()
            
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            p_inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]
                        }
            
            q_inputs = {'input_ids': batch[3],
                        'attention_mask': batch[4],
                        'token_type_ids': batch[5]}

            #outputs with similiarity score
            outputs = model(p_inputs,q_inputs)

            # target: position of positive samples = diagonal element 
            targets = torch.arange(0, args.per_device_train_batch_size).long()
            if torch.cuda.is_available():
                targets = targets.to('cuda')

            sim_scores = F.log_softmax(outputs, dim=1)

            loss = F.nll_loss(sim_scores, targets)
            #print(loss)
            total_loss+=loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            
            torch.cuda.empty_cache()
        torch.save(model.state_dict(), f'/opt/ml/input/code/colbert/best_model/colbert_epoch{epoch+1}.pth')
        print(total_loss/steps)

        
    return model
    


if __name__ == '__main__':
    main()
