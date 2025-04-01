import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset, random_split
from my_dataset import TranslationDataset
from data_preparation import load_translation_dataset, prepare_data_with_hf

def prepare_dataset(subset_size, tokenizer):
    dataset = load_translation_dataset()
    processed_data, tokenizer = prepare_data_with_hf(dataset)
    full_dataset = TranslationDataset(processed_data)
    
    indices = range(min(subset_size, len(full_dataset)))
    subset = Subset(full_dataset, indices)
    
    train_size = int(0.7 * len(subset))
    val_size = int(0.15 * len(subset))
    test_size = len(subset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        subset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset, test_dataset, tokenizer

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader