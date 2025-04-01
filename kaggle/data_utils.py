import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset, random_split
from .dataset import TranslationDataset
from .data_preparation import load_translation_dataset, prepare_data_with_hf

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

def translate_sentence(model, tokenizer, sentence, device, max_length=128):
    model.eval()
    
    inputs = tokenizer(
        f"{tokenizer.bos_token} {sentence}",
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    decoder_input = torch.tensor([[tokenizer.bos_token_id]], device=device)
    
    with torch.no_grad():
        for _ in range(max_length):
            src_mask = (inputs['input_ids'] != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
            tgt_mask = torch.tril(torch.ones(decoder_input.size(1), decoder_input.size(1), device=device)).bool()
            
            output = model(
                inputs['input_ids'],
                decoder_input,
                src_mask,
                tgt_mask.unsqueeze(0).unsqueeze(0)
            )
            
            next_token = output[:, -1].argmax(-1)
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(decoder_input[0], skip_special_tokens=True)