import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def load_translation_dataset():
    print("Loading Tatoeba en-ru...")
    try:
        dataset = load_dataset("Helsinki-NLP/tatoeba", lang1="en", lang2="ru", trust_remote_code=True)
        
    except Exception as e:
        print(f"Error while loading dataset: {e}")
        raise
    
    print("\nDataset structure:")
    print(dataset)
    
    print("\nData sample:")
    for i in range(2):
        print(f"EN: {dataset['train'][i]['translation']['en']}")
        print(f"RU: {dataset['train'][i]['translation']['ru']}\n")

    return dataset

def prepare_data_with_hf(
    dataset, 
    model_name: str = "Helsinki-NLP/opus-mt-en-ru", 
    max_length: int = 128, 
    batch_size: int = 32
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    special_tokens = {
        'bos_token': '<s>',
        'eos_token': '</s>',
        'pad_token': '<pad>',
        'unk_token': '<unk>'
    }
    tokenizer.add_special_tokens(special_tokens)
    
    tokenizer.bos_token = '<s>'
    tokenizer.eos_token = '</s>'
    tokenizer.pad_token = '<pad>'
    tokenizer.unk_token = '<unk>'
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids('<s>')
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('</s>')
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')

    def preprocess_function(examples):
        source_texts = [
            f"{tokenizer.bos_token} {item['en']}" 
            for item in examples['translation']
        ]
        
        target_texts = [
            f"{tokenizer.bos_token} {item['ru']} {tokenizer.eos_token}"
            for item in examples['translation']
        ]

        source_encoding = tokenizer(
            source_texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='np',
            add_special_tokens=False 
        )
        
        target_encoding = tokenizer(
            target_texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='np',
            add_special_tokens=False
        )

        source_ids = source_encoding['input_ids']
        source_ids[:, 0] = tokenizer.bos_token_id 
        
        target_ids = target_encoding['input_ids']
        target_ids[:, 0] = tokenizer.bos_token_id 
        
        target_attention_mask = target_encoding['attention_mask']
        seq_lens = np.argmin(target_attention_mask, axis=1)
        seq_lens[seq_lens == 0] = target_attention_mask.shape[1]
        
        for i, pos in enumerate(seq_lens):
            if pos < max_length:
                target_ids[i, pos-1] = tokenizer.eos_token_id
            else:
                target_ids[i, -1] = tokenizer.eos_token_id

        return {
            'input_ids': source_ids,
            'attention_mask': source_encoding['attention_mask'],
            'labels': target_ids,
            'decoder_attention_mask': target_encoding['attention_mask']
        }
    
    processed_dataset = dataset['train'].map(
        preprocess_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset['train'].column_names
    )
    
    return processed_dataset, tokenizer

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
