import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification
from tqdm import tqdm
import json
import numpy as np
import argparse
import random

# Argument parser for hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--ppo_epochs', type=int, default=50, help='Number of PPO epochs')
parser.add_argument('--lr', type=float, default=9e-6, help='Learning rate')
parser.add_argument('--beta', type=float, default=0.02, help='Beta for KL penalty')
# parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--ppo_clip_ratio', type=float, default=0.2, help='PPO clip ratio')
# parser.add_argument('--mini_batch_size', type=int, default=64, help='Mini-batch size')
parser.add_argument('--mini_batch_size', type=int, default=4, help='Mini-batch size')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

# Set the seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained reward model
reward_model = GPT2ForSequenceClassification.from_pretrained("models/reward_model", num_labels=1).to(device)
reward_model.eval()
reward_model.config.pad_token_id = reward_model.config.eos_token_id
reward_model = torch.nn.DataParallel(reward_model)

# Load the supervised fine-tuned model
sft_model = GPT2LMHeadModel.from_pretrained("models/sft_model/sft_model_gpt2_with_original_data").to(device)
sft_model.eval()
sft_model = torch.nn.DataParallel(sft_model)

# Load the ppo model from supervised fine-tuned model
ppo_model = GPT2LMHeadModel.from_pretrained("models/sft_model/sft_model_gpt2_with_original_data").to(device)
ppo_model.train()
ppo_model = torch.nn.DataParallel(ppo_model)

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


# Functions to generate text given the instructions
def generate_text(model, instruction, max_length=1023):
    '''
    Generate text from the model given the input text
    Input:
        model: GPT2LMHeadModel
        text: str
        max_new_tokens: int
    Output:
        inputs: torch.tensor, the token ids of the instruction
        outputs: torch.tensor, the token ids of the instruction with generated demonstration
    '''
    encoded_inputs = tokenizer.encode_plus(instruction, return_tensors='pt', max_length=max_length, truncation=True)
    inputs = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)
    outputs = model.module.generate(inputs, attention_mask=attention_mask, max_length=max_length, do_sample=True, pad_token_id=tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id, temperature=1.0)
    # outputs = model.generate(inputs, attention_mask=attention_mask, max_length=max_length, do_sample=True, pad_token_id=tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id, temperature=1.0)
    return [inputs, outputs]


# Functions to calculate the log probability of demonstration (interaction subtracts instruction) given instruction
def get_log_prob(model, instruction_ids, interaction_ids):
    '''
    Calculate the log probability of demonstration (interaction subtracts instruction) given instruction
    Input:
        model: GPT2LMHeadModel
        instruction_ids: torch.tensor, the token ids of the instruction
        interaction_ids: torch.tensor, the token ids of the instruction with generated demonstration
    Output:
        log_prob: float, the log probability of demonstration (interaction subtracts instruction) given instruction
    '''
    # Calculate the number of tokens in the instruction and interaction
    instruction_len = instruction_ids.size(1)
    interaction_len = interaction_ids.size(1)

    # Get the output logits from the model
    outputs = model.module(interaction_ids)
    # outputs = model(interaction_ids)
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


# Prepare the data
class DialogueDataset(Dataset):
    def __init__(self, path):
        with open(path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        interaction = self.data[index]['chat']
        instruction, _ = interaction.rsplit('\n\nAssistant: ', 1)
        instruction += '\n\nAssistant: '
        return instruction


# Initialize dataloader
train_dataset = DialogueDataset('data/filter_gen_dataset_mhy_train.json')
val_dataset = DialogueDataset('data/filter_gen_dataset_mhy_val.json')

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)


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
    padded_sequences = torch.ones(batch_size, max_length, dtype=torch.long).to(device) * padding_value
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq[0])] = seq[0]
    return padded_sequences


# Functions to calculate the rewards
def calculate_rewards(model, interactions):
    '''
    Calculate the rewards for the given interactions
    Input:
        model: GPT2ForSequenceClassification
        interactions: list of torch.tensor, the token ids of the interactions
    Output:
        rewards: torch.tensor, the rewards for the given interactions
    '''
    # Pad all interactions to 1024.
    interactions = pad_sequence(interactions, max_length=1024, padding_value=tokenizer.pad_token_id)
    with torch.no_grad():
        rewards = model.module(interactions).logits.squeeze()
    return rewards


def ppo_iter(mini_batch_size, instructions_interactions, advantage):
    '''
    Generate mini batches for PPO update
    Input:
        mini_batch_size: int
        instructions_interactions: list of tuple of torch.tensor, the token ids of the instructions and interactions. The length of the list is the batch size
        advantage: torch.tensor, the advantage of the interactions, the shape is (batch_size, 1)
    Output:
        tuple of two lists and one torch.tensor, the token ids of the instructions, interactions and the advantage
    '''
    batch_size = advantage.size(0)
    ids = np.random.permutation(batch_size)
    ids = ids[:mini_batch_size * (batch_size // mini_batch_size)]

    for i in range(0, len(ids), mini_batch_size):
        batch_ids = ids[i:i + mini_batch_size]
        yield [instructions_interactions[i][0] for i in batch_ids], [instructions_interactions[i][1] for i in batch_ids], advantage[batch_ids]


def ppo_update(mini_batch_size, instructions_interactions, advantages, clip_param=args.ppo_clip_ratio):
    for instructions, interactions, advantage in ppo_iter(mini_batch_size, instructions_interactions, advantages):
        
        # Calculate the log probability of the generated demonstrations
        ppo_log_probs = [get_log_prob(ppo_model, inst, inter) for inst, inter in zip(instructions, interactions)]
        with torch.no_grad():
            sft_log_probs = [get_log_prob(sft_model, inst, inter) for inst, inter in zip(instructions, interactions)]
            
        # Convert log probabilities to tensors
        new_log_probs = torch.stack(ppo_log_probs)
        old_log_probs = torch.stack(sft_log_probs)
        old_log_probs = old_log_probs.detach()
        
        ratio = (new_log_probs - old_log_probs).exp()
        
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

        loss  = - torch.min(surr1, surr2).mean()
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()


optimizer = torch.optim.Adam(ppo_model.parameters(), lr=args.lr)

best_reward = -float('inf')
for epoch in range(args.ppo_epochs):
    for instructions in tqdm(train_dataloader):
        # Generate demonstrations from the ppo model given the input instruction
        # instructions_interactions is a list of [instruction_ids, interaction_ids]
        with torch.no_grad():
            instructions_interactions = [generate_text(ppo_model, i) for i in instructions]
            interactions = [i[1] for i in instructions_interactions]

            # Calculate the reward given interactions
            rewards = calculate_rewards(reward_model, interactions)

        # Calculate the advantages
        advantages = rewards - rewards.mean()
        advantages = advantages.reshape(-1, 1)

        # Update the model
        ppo_update(args.mini_batch_size, instructions_interactions, advantages)
        
    # Evaluation loop
    with torch.no_grad():
        avg_reward = 0
        num_batches = 0
        for instruction in val_dataloader:
            # Generate demonstrations from the ppo model given the input instruction
            instructions_interactions = [generate_text(ppo_model, i) for i in instructions]
            interactions = [i[1] for i in instructions_interactions]
            
            # Calculate the reward given interactions
            rewards = calculate_rewards(reward_model, interactions)
            
            avg_reward += rewards.mean().item()
            num_batches += 1
        avg_reward /= num_batches
        print('Epoch {}, average reward {}'.format(epoch, avg_reward))
        
        # Save the model if the average reward is higher than the previous best
        if avg_reward > best_reward:
            best_reward = avg_reward
            ppo_model.save_pretrained('models/ppo_model')
