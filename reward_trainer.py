import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW
from src.dataset import GroupedComparisonDataset
from tqdm import tqdm
import json
from torch.utils.checkpoint import checkpoint
import argparse

MODEL_NAME = 'gpt2'


def collate_fn(batch):
    # batch is a list of group data from __getitem__
    # We want to keep input_ids and attention_mask as lists because each group can have different number of interactions
    return {'input_ids': [item['input_ids'] for item in batch],
            'attention_mask': [item['attention_mask'] for item in batch],
            'confidence': [item['confidence'] for item in batch],
            'sol_id': [item['sol_id'] for item in batch]}
    
    
def evaluate(eval_loader, model, device, batch_size):
    '''
    Evaluate the trained model.

    INPUT: 
      - eval_dataset: dataset for evaluation
      - model: trained model
      - device: hardware device for training ('cpu' or 'cuda')
      - batch_size: batch size for load the dataset
    '''
    
    model.eval()
    eval_loss = 0
    num_eval_batches = 0
    
    with torch.no_grad():
        for eval_group_batch in tqdm(eval_loader, total=len(eval_loader)):
            eval_batch_loss = 0
            eval_num_groups = 0
            for eval_group in zip(eval_group_batch['input_ids'], eval_group_batch['attention_mask'], eval_group_batch['confidence']):
                eval_input_ids = [entry.to(device).view(1, -1) for entry in eval_group[0]]
                eval_attention_mask = [entry.to(device).view(1, -1) for entry in eval_group[1]]
                eval_confidence = eval_group[2]
                
                eval_group_loss = 0
                eval_num_comparisons = 0
                
                # Compute rewards for all interactions in the validation group
                eval_rewards = [model(entry_ids, attention_mask=am).logits for entry_ids, am in zip(eval_input_ids, eval_attention_mask)]
                
                # Compute comparisons within the validation group
                for i in range(len(eval_rewards)):
                    for j in range(i + 1, len(eval_rewards)):
                        if eval_confidence[i] != eval_confidence[j]: 
                            eval_winner = i if eval_confidence[i] > eval_confidence[j] else j
                            eval_loser = j if eval_winner == i else i
                            eval_group_loss -= torch.log(torch.sigmoid(eval_rewards[eval_winner] - eval_rewards[eval_loser]))
                            eval_num_comparisons += 1
                            
                if eval_num_comparisons > 0:
                    eval_group_loss /= eval_num_comparisons
                    eval_batch_loss += eval_group_loss
                    eval_num_groups += 1
                    
            if eval_num_groups > 0:
                eval_batch_loss /= eval_num_groups
                eval_loss += eval_batch_loss.item()
                num_eval_batches += 1
                
    model.train()
    return eval_loss / num_eval_batches if num_eval_batches > 0 else 0
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_pairs_path', type=str, default='data/analysis/pretrain_pairs.json')
    parser.add_argument('--finetune_pairs_path', type=str, default='data/analysis/finetune_pairs.json')
    parser.add_argument('--save_path', type=str, default='model/finetuned_model/')
    parser.add_argument('--train_dataset_path', type=str, default='data/rw/dataset_train.json')
    parser.add_argument('--val_dataset_path', type=str, default='data/rw/dataset_val.json')
    parser.add_argument('--test_dataset_path', type=str, default='data/test/dataset_full.json')
    args = parser.parse_args()
    
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)
    
    train_dataset = GroupedComparisonDataset(args.train_dataset_path, tokenizer, max_length=1024)
    # train_dataset = GroupedComparisonDataset('data/fake_dataset.json', tokenizer, max_length=1024)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)  # each batch will contain 8 groups
    eval_dataset = GroupedComparisonDataset(args.val_dataset_path, tokenizer, max_length=1024)
    # eval_dataset = GroupedComparisonDataset('data/fake_validation_dataset.json', tokenizer, max_length=1024)
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn) # each batch will contain 8 groups
    test_dataset = GroupedComparisonDataset(args.test_dataset_path, tokenizer, max_length=1024)
    
    # save the score-confidence pairs of the pretrained reward model
    # pairs_path = args.pretrain_pairs_path
    # pretrain_pairs = []
    # with torch.no_grad():
    #     for group in tqdm(test_dataset, total=len(test_dataset)):
    #         pretrain_input_ids = [entry.to(device).view(1, -1) for entry in group['input_ids']]
    #         pretrain_attention_mask = [entry.to(device).view(1, -1) for entry in group['attention_mask']]
    #         test_rewards = [model(entry_ids, attention_mask=am).logits for entry_ids, am in zip(pretrain_input_ids, pretrain_attention_mask)]
    #         group['scores'] = [float(reward) for reward in test_rewards]
    #         group_dict = {'confidence': group['confidence'], 'scores': group['scores'], 'sol_id': group['sol_id']}
    #         pretrain_pairs.append(group_dict)
                
    #     with open(pairs_path, "w") as f:
    #         json.dump(pretrain_pairs, f)
            
    optimizer = AdamW(model.parameters(), lr=9e-6, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=15*len(train_loader), eta_min=9e-6*0.1)  # Assuming one epoch

    best_eval_loss = 100
    save_path = args.save_path
    model.zero_grad()
    model.train()
    training_step = 0
    for epoch in range(1):  # train for one epoch
        epoch_loss = 0
        num_comparisons = 0

        for group_batch in tqdm(train_loader, total=len(train_loader)):
            for group in zip(group_batch['input_ids'], group_batch['attention_mask'], group_batch['confidence']):
                input_ids = [entry.to(device).view(1, -1) for entry in group[0]]
                attention_mask = [entry.to(device).view(1, -1) for entry in group[1]]
                confidence = group[2]
                
                # Compute comparisons within the group
                for i in range(len(input_ids)):
                    for j in range(i + 1, len(input_ids)):
                        if confidence[i] != confidence[j]:  # if confidence scores are not equal
                            optimizer.zero_grad()
                            winner_idx = i if confidence[i] > confidence[j] else j
                            loser_idx = j if winner_idx == i else i

                            winner_input_ids = input_ids[winner_idx]
                            loser_input_ids = input_ids[loser_idx]

                            winner_attention_mask = attention_mask[winner_idx]
                            loser_attention_mask = attention_mask[loser_idx]

                            winner_logits = model(winner_input_ids, attention_mask=winner_attention_mask).logits
                            loser_logits = model(loser_input_ids, attention_mask=loser_attention_mask).logits

                            comparison_loss = -torch.log(torch.sigmoid(winner_logits - loser_logits))
                            comparison_loss.backward()
                            optimizer.step()
                            scheduler.step()
                            training_step += 1
                            epoch_loss += comparison_loss.item()
                            num_comparisons += 1
                
                        if training_step % 200 == 0:
                            eval_loss = evaluate(eval_loader, model, device, 8)
                            print(f"Eval loss: {eval_loss}")

        train_loss = epoch_loss / num_comparisons if num_comparisons > 0 else 0    
        eval_loss = evaluate(eval_loader, model, device, 8)
        print(f'Epoch: {epoch} | Training Loss: {train_loss} | Validation Loss: {eval_loss}')
        
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print("Model Saved!")
                
    # save the score-confidence pairs of the trained reward model
    model = GPT2ForSequenceClassification.from_pretrained(save_path, num_labels=1).to(device)

    pairs_path = args.finetune_pairs_path
    train_pairs = []
    with torch.no_grad():
        for group in tqdm(test_dataset, total=len(test_dataset)):
            train_input_ids = [entry.to(device).view(1, -1) for entry in group['input_ids']]
            train_attention_mask = [entry.to(device).view(1, -1) for entry in group['attention_mask']]
            test_rewards = [model(entry_ids, attention_mask=am).logits for entry_ids, am in zip(train_input_ids, train_attention_mask)]
            group['scores'] = [float(reward) for reward in test_rewards]
            group_dict = {'confidence': group['confidence'], 'scores': group['scores'], 'sol_id': group['sol_id']}
            train_pairs.append(group_dict)
            
        with open(pairs_path, "w") as f:
            json.dump(train_pairs, f)
