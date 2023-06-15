#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Project       : Modern NLP Milestone 3
@File          : ppo_trainer.py
@Author        : Yiyang Feng
@Date          : 2023/06/15 11:55
@Version       : 1.0
"""
import os
import torch
from src.dialogue import DialogueDataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification
from tqdm import tqdm
import json
import numpy as np
import argparse
import random


class PPOTrainer(object):
    def __init__(self, args, train_dataloader, val_dataloader):
        super(PPOTrainer, self).__init__()
        # Set the args
        self.args = args

        # Set the device
        self.device_ppo = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_sft = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.device_rm  = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        # Load the pre-trained reward model
        self.reward_model = GPT2ForSequenceClassification.from_pretrained("models/reward_model", num_labels=1).to(self.device_rm)
        self.reward_model.eval()
        self.reward_model.config.pad_token_id = self.reward_model.config.eos_token_id

        # Load the supervised fine-tuned model
        # sft_model = GPT2LMHeadModel.from_pretrained("models/sft_model/sft_model_gpt2_with_original_data").to(device)
        self.sft_model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(self.device_sft)
        self.sft_model.eval()

        # Load the ppo model from supervised fine-tuned model
        # ppo_model = GPT2LMHeadModel.from_pretrained("models/sft_model/sft_model_gpt2_with_original_data").to(device)
        self.ppo_model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(self.device_ppo)
        self.ppo_model.train()

        # Load the tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Define the maximum length of the input text
        self.rm_max_length = 1024
        self.lm_max_length = 1024
        
        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.ppo_model.parameters(), lr=self.args.lr)
        
        # Define the dataloader
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def generate_text(self, model, instruction, max_length=1023, mode="ppo"):
        '''
        Generate text from the model given the input text
        Input:
            model: GPT2LMHeadModel
            instruction: str
            max_length: int, the total length of the input text and generated text
            mode: str, "ppo" or "sft"
        Output:
            inputs: torch.tensor, the token ids of the instruction
            outputs: torch.tensor, the token ids of the instruction with generated demonstration
        '''
        # Assert the mode
        assert mode in ["ppo", "sft"], "mode must be either 'ppo' or 'sft'"
        
        # Move the inputs to the device
        if mode == "ppo":
            device = self.device_ppo
        else:
            device = self.device_sft    
        encoded_inputs = self.tokenizer.encode_plus(instruction, return_tensors='pt', max_length=max_length, truncation=True)
        inputs = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)
        
        try:
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id = self.tokenizer.eos_token_id,
                temperature=1.0,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        except:
            print('Error in generating text')
            print('Instruction: ', instruction)
            print('Input: ', inputs)
            print('Input shape: ', inputs.shape)
            print('Attention mask: ', attention_mask)
            print('Max length: ', max_length)
            raise
        return [inputs, outputs]

    def get_log_prob(self, model, instruction_ids, interaction_ids, mode="ppo"):
        '''
        Calculate the log probability of demonstration (interaction subtracts instruction) given instruction
        Input:
            model: GPT2LMHeadModel
            instruction_ids: torch.tensor, the token ids of the instruction
            interaction_ids: torch.tensor, the token ids of the instruction with generated demonstration
            mode: str, "ppo" or "sft"
        Output:
            log_prob: float, the log probability of demonstration (interaction subtracts instruction) given instruction
        '''
        # Assert the mode
        assert mode in ["ppo", "sft"], "mode must be either 'ppo' or 'sft'"
        
        # Move the inputs to the device
        if mode == "ppo":
            device = self.device_ppo
        else:
            device = self.device_sft
        interaction_ids = interaction_ids.to(device)
        instruction_ids = instruction_ids.to(device)
        
        # Calculate the number of tokens in the instruction and interaction
        instruction_len = instruction_ids.size(1)
        interaction_len = interaction_ids.size(1)

        # Get the output logits from the model
        outputs = model(interaction_ids)
        logits = outputs.logits[:, :-1, :]  # Exclude the last token

        # Get the softmax probabilities from the logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Get the probabilities of the actual tokens
        actual_probabilities = probabilities.gather(2, interaction_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # Exclude the first token

        # Only consider the probabilities of the demonstration tokens (exclude the tokens of the instruction)
        demonstration_probabilities = actual_probabilities[:, instruction_len:]

        # Take the log of the probabilities and sum them up to get the log probability of the entire sequence
        log_prob = torch.log(demonstration_probabilities).sum(dim=-1)

        return log_prob

    def calculate_rewards(self, model, interactions):
        '''
        Calculate the rewards for the given interactions
        Input:
            model: GPT2ForSequenceClassification
            interactions: list of torch.tensor, the token ids of the interactions
        Output:
            rewards: torch.tensor, the rewards for the given interactions
        '''
        
        # Pad all interactions to the max length of the reward model.
        def pad_sequence(sequences, max_length=1024, padding_value=0):
            '''
            Pad a list of variable length Tensors with padding_value to the max length
            Input:
                sequences: list of torch.tensor with shape (1, seq_len). The length of the list is the batch size
                max_length: int
                padding_value: int
            Output:
                torch.tensor, the padded tensor
            '''
            batch_size = len(sequences)
            padded_sequences = torch.ones(batch_size, max_length, dtype=torch.long) * padding_value
            for i, seq in enumerate(sequences):
                padded_sequences[i, :len(seq[0])] = seq[0]
            return padded_sequences
        
        interactions = pad_sequence(interactions, max_length=self.rm_max_length, padding_value=self.tokenizer.pad_token_id).to(self.device_rm)
        
        with torch.no_grad():
            rewards = model(interactions).logits.squeeze()
        return rewards

    def ppo_update(self, mini_batch_size, instructions_interactions, rewards, clip_param=0.2, beta=0.02):
        '''
        Update the PPO model
        Input:
            mini_batch_size: int
            instructions_interactions: list of tuple of torch.tensor, the token ids of the instructions and interactions. The length of the list is the batch size
            rewards: torch.tensor, the reward of the interactions, the shape is (batch_size, 1)
            clip_param: float, the clip parameter for PPO
            beta: float, the entropy coefficient
        '''
        # Get the token ids of the instructions and interactions and rewards
        def ppo_iter(mini_batch_size, instructions_interactions, reward):
            '''
            Generate mini batches for PPO update
            Input:
                mini_batch_size: int
                instructions_interactions: list of tuple of torch.tensor, the token ids of the instructions and interactions. The length of the list is the batch size
                reward: torch.tensor, the reward of the interactions, the shape is (batch_size, 1)
            Output:
                tuple of two lists and one torch.tensor, the token ids of the instructions, interactions and the reward
            '''
            batch_size = reward.size(0)
            ids = np.random.permutation(batch_size)
            ids = ids[:mini_batch_size * (batch_size // mini_batch_size)]

            for i in range(0, len(ids), mini_batch_size):
                batch_ids = ids[i:i + mini_batch_size]
                yield [instructions_interactions[i][0] for i in batch_ids], [instructions_interactions[i][1] for i in batch_ids], reward[batch_ids]
                
        for instructions, interactions, reward in ppo_iter(mini_batch_size, instructions_interactions, rewards):
            
            # Calculate the log probability of the generated demonstrations
            ppo_log_probs = [self.get_log_prob(self.ppo_model, inst, inter, mode="ppo") for inst, inter in zip(instructions, interactions)]
            with torch.no_grad():
                sft_log_probs = [self.get_log_prob(self.sft_model, inst, inter, mode="sft") for inst, inter in zip(instructions, interactions)]
                
            # Convert log probabilities to tensors
            new_log_probs = torch.stack(ppo_log_probs)
            old_log_probs = torch.stack(sft_log_probs)
            old_log_probs = old_log_probs.detach()
            
            # Move to the same device
            new_log_probs = new_log_probs.to(self.device_ppo)
            old_log_probs = old_log_probs.to(self.device_ppo)
            reward = reward.to(self.device_ppo)
            
            ratio = (new_log_probs - old_log_probs).exp().clamp(-10, 10)
            
            R = reward - beta * (new_log_probs - old_log_probs)
            advantage = R - R.mean()
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            loss  = - torch.min(surr1, surr2).mean()
            print('Ratio: ', ratio)
            print('surr1: ', surr1)
            print('surr2: ', surr2)
            print('advantage: ', advantage)
            print('loss: ', loss)
            self.optimizer.zero_grad()
            
            loss.backward()
            self.optimizer.step()

    def train(self):
        '''
        Train the model
        '''
        best_reward = -float('inf')
        for epoch in range(self.args.ppo_epochs):
            for instructions, _, _ in tqdm(self.train_dataloader):
                # Generate demonstrations from the ppo model given the input instruction
                # instructions_interactions is a list of [instruction_ids, interaction_ids]
                with torch.no_grad():
                    instructions_interactions = [self.generate_text(self.ppo_model, i, max_length=self.lm_max_length-1, mode="ppo") for i in instructions]
                    interactions = [i[1] for i in instructions_interactions]

                    # Calculate the reward given interactions
                    rewards = self.calculate_rewards(self.reward_model, interactions)
                    print('rewards: ', rewards)

                # Reshape the rewards
                rewards = rewards.reshape(-1, 1)

                # Update the model
                self.ppo_update(self.args.mini_batch_size, instructions_interactions, rewards, clip_param=self.args.ppo_clip_ratio, beta = self.args.beta)
                
            # Evaluation loop
            with torch.no_grad():
                avg_reward = 0
                num_batches = 0
                for instruction, _, _ in self.val_dataloader:
                    # Generate demonstrations from the ppo model given the input instruction
                    instructions_interactions = [self.generate_text(self.ppo_model, i, max_length=self.lm_max_length-1, mode="ppo") for i in instructions]
                    interactions = [i[1] for i in instructions_interactions]
                    
                    # Calculate the reward given interactions
                    rewards = self.calculate_rewards(self.reward_model, interactions)
                    
                    avg_reward += rewards.mean().item()
                    num_batches += 1
                avg_reward /= num_batches
                print('Epoch {}, average reward {}'.format(epoch, avg_reward))
                
                # Save the model if the average reward is higher than the previous best
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.ppo_model.save_pretrained('models/ppo_model')


if __name__ == '__main__':
    # Argument parser for hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppo_epochs', type=int, default=50, help='Number of PPO epochs')
    parser.add_argument('--lr', type=float, default=9e-6, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.02, help='Beta for KL penalty')
    # parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--ppo_clip_ratio', type=float, default=0.2, help='PPO clip ratio')
    # parser.add_argument('--mini_batch_size', type=int, default=64, help='Mini-batch size')
    parser.add_argument('--mini_batch_size', type=int, default=2, help='Mini-batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--augment', action='store_true', default=True, help='Augment the data')
    args = parser.parse_args()

    # Set the seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    # Initialize dataloader
    train_dataset = DialogueDataset('data/gen_dataset_mhy_train.json', augmentation=args.augment)
    val_dataset = DialogueDataset('data/gen_dataset_mhy_val.json', augmentation=args.augment)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize the PPO trainer
    trainer = PPOTrainer(args, train_dataloader, val_dataloader)
    trainer.train()
