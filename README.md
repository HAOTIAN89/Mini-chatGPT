[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/YM0Aj0xh)
# Mini-GPTA: <u>Mini</u> (Distilled) Chat<u>GPT</u>eaching <u>A</u>ssistant

This is the project repository for the final report [*Mini-GPTA: Distilled ChatGPTeaching Assistant for EPFL Courses*](https://github.com/CS-552/project-m3-mhy/blob/main/final_report_mhy.pdf) by MHY Team in EPFL Modern NLP course (CS-552).

## MHY-Team Members

- Yiyang Feng: yiyang.feng@epfl.ch
- Haotian Wu: haotian.wu@epfl.ch
- Maciej Styczen: maciej.styczen@epfl.ch

## Abstract

In recent years, the development of autonomous AI tutors for educational purposes has attracted substantial attention. However, current methodologies either require large volumes of human-labeled data for training smaller models or pose challenges due to computational resources and closed-source issues when using large language models. This paper introduces Mini-GPTA: <u>Mini</u>-<u>GPT</u>eaching <u>A</u>ssistant, a specialized chatbot targeting course content for EPFL engineering students, which is a GPT-2-medium model distilled from ChatGPT. Through interaction with ChatGPT, we obtain human-machine dialogue data and augment it with external datasets. These data then feed into our instruction tuning and Reinforcement Learning with Human Feedback (RLHF) process, resulting in a small yet efficient model. Mini-GPTA outperforms baseline methods by a considerable margin, achieving up to 10 times higher similarity with ground truth, demonstrating the effectiveness of our approach. This work contributes a potential pathway for achieving resource-efficient, specialized educational domain chatbots, sparking new opportunities for individual researchers and smaller institutions to create high-performing.

## Install requirements

```bash
conda create -n nlp python=3.9.16
conda activate nlp
pip install -r requirements.txt
```

## Reproducibility

### Reward Model Training

```bash
python reward_trainer.py
```

### Data Processing for Supervised Fine-tuning and PPO Models

Execute the cells of `process_data.ipynb`.

### Supervised Fine-tuned Model Training

Execute the cells of `sft_trainer.ipynb`.

### PPO Model Training

```bash
python ppo_trainer.py
```

### Evaluation

For semantic scores, execute the cells of `evaluation.ipynb`.

For answers generation, execute `python gen_script_mhy.py`.

## Files

- `data/`: the folder for storing our data.
    - `data/m2/*`: original datasets from milestone 2.
    - `data/rw/*`: the training and validation datasets for reward model.
    - `data/ppo/*`: the training and validation datasets for ppo model.
    - `data/sft/*`: the training and validation datasets for sft model.
    - `data/prompts.json`: the sample JSON of the testing questions provided by TAs.
- `models/`: the folder for storing our models.
    - `models/reward_model`: finetuned reward model.
    - `models/sft_model`: supervised fine-tuned model with different settings.
    - `models/ppo_model`: PPO model.
- `src/dialogue.py`: the dataset for dataloader of training sft model process.
- `answers_mhy_evaluation/`: the folder for generated answers of `prompts.json` from different models, including:
    - `answers_mhy_evaluation/answers_mhy_ppo_model.jsonl`.
    - `answers_mhy_evaluation/answers_mhy_pretrained_gpt2_medium.jsonl`.
    - `answers_mhy_evaluation/answers_mhy_sft_model_gpt2_medium_with_augmentation_data.jsonl`.
    - `answers_mhy_evaluation/answers_mhy_sft_model_gpt2_medium_with_original_data.jsonl`.
- `answers_mhy.json`: the selected json file of answers generated by our generative model for the testing questions in `prompts.json` for grading.
- `evaluation.ipynb`: the jupyter notebook file for the evaluation of our educational chatbots.
- `gen_script_mhy.py`: the python script allowing anyone to easily access our generative model to produce answers for the testing questions.
- `reward_trainer`: the script to train the reward model.
- `sft_trainer.ipynb`: the jupyter notebook file for training and validation processing of sft models.
- `ppo_trainer.py`: the python script for training and validation processing of ppo model.
- `process_data.ipynb`: the jupyter notebook file for cleaning up and changing the format of our external datasets.
- `final_report_mhy.pdf`: modern NLP project final report.
- `python.txt`: the txt file specifying the python versions.
- `requirements.txt`: the txt file specifying the package versions.
