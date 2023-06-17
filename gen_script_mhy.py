from pyexpat import model
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_cosine_schedule_with_warmup, GPT2ForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import nltk
from rouge_score import rouge_scorer
import json
import os


# constants
ASSISTANT_MODEL_PATH = "./models/sft_model/sft_model_gpt2_medium_with_augmentation_data"
TEST_DATASET_PATH = "data/prompts.json"
OUTPUT_PATH = "answers_mhy.jsonl"
MAX_LEN = 1024
NUM_BEAMS =  5
NUM_RETURNED_SEQUENCES = 5


# load the pre-trained assistant model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assistant_tokenizer = GPT2Tokenizer.from_pretrained(ASSISTANT_MODEL_PATH)
assistant_model = GPT2LMHeadModel.from_pretrained(ASSISTANT_MODEL_PATH).to(device)
assistant_model.eval()


def prepare_model_input(datapoint):
    # if a multiple choice question, add the choices to the model input
    question_body = "User: " + datapoint["question"]

    if "choices" in datapoint and datapoint["choices"] not in [None, []] :
        question_body += ("\n" + "Options:\n" + "\n".join(datapoint["choices"]))
    
    return question_body + "\nAssistant: "

def assistant_query(question):
    # Define the input question
    input_ids = assistant_tokenizer.encode(question, return_tensors='pt').to(device)

    # Generate the attention mask
    attention_mask = torch.ones_like(input_ids).to(device)

    # Generate the answers
    output = assistant_model.generate(input_ids=input_ids, attention_mask = attention_mask, max_length=MAX_LEN, num_beams=NUM_BEAMS, no_repeat_ngram_size=2, num_return_sequences=NUM_RETURNED_SEQUENCES, early_stopping=True)
    
     # Decode the answers, return first one that is non-empty
    for output_sentence in output:
        decoded_output_sentence = assistant_tokenizer.decode(output_sentence, skip_special_tokens=True)
        if decoded_output_sentence != "":
            return decoded_output_sentence

    # if no non-empty response found, return empty string
    return ""

# load the dataset
with open(TEST_DATASET_PATH, "r") as f:
    test_dataset = json.load(f)

# verify which models 
done_ids = []
with open(OUTPUT_PATH, mode='r') as f:
    for line in f.readlines():
        data = json.loads(line.strip())
        done_ids.append(data["guid"])

# for each datapoint, generate an answer - save to jsonl with append mode (allows for stopping the script while saving partial results)
with open(OUTPUT_PATH, "a") as f:
    for i, datapoint in enumerate(test_dataset):
        question_id = datapoint["guid"]
        if question_id not in done_ids:
            print(f"Processing question: {question_id} - {i}/{len(test_dataset)} done")
            question = prepare_model_input(datapoint)
            print("Question: ", question)
            model_answer = assistant_query(question)
            print("Answer: ", model_answer)
            f.write(json.dumps({
                "guid": question_id,
                "answer": model_answer
            }))


# convert jsonl to json
predictions = []
with open(OUTPUT_PATH, mode='r') as f:
    for line in f.readlines():
        predictions.append(json.loads(line.strip()))

with open("answers_mhy.json", "w") as f:
    json.dump(predictions, f)
        