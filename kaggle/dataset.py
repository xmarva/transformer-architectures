from torch.utils.data import Dataset
import torch

class TranslationDataset(Dataset):
    def __init__(self, processed_data):
        self.input_ids = processed_data['input_ids']
        self.attention_mask = processed_data['attention_mask']
        self.labels = processed_data['labels']
        self.decoder_attention_mask = processed_data['decoder_attention_mask']
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'decoder_attention_mask': torch.tensor(self.decoder_attention_mask[idx], dtype=torch.long)
        }