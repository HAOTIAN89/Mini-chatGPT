import json
from torch.utils.data import Dataset


class DialogueDataset(Dataset):
    '''
    Dataset for dialogue data
    '''
    def __init__(self, filename, augmentation=True):
        with open(filename, 'r') as f:
            raw_data = json.load(f)
            
        if augmentation:
            with open('data/gen_dataset_mhy_math.json', 'r') as f1:
                math_data = json.load(f1)
            with open('data/gen_dataset_mhy_openbookqa.json', 'r') as f2:
                code_data = json.load(f2) 
            raw_data.extend(math_data)
            raw_data.extend(code_data)
            self.data = raw_data
        else:
            self.data = raw_data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chat = self.data[idx]['chat']
        instruction, demonstration = chat.rsplit('Assistant: ', 1)
        instruction = instruction + 'Assistant: '
            
        return instruction, demonstration, chat