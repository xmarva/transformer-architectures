import yaml
from src.data.dataset_factory import DatasetFactory
from src.data.tokenization.bpe import BPETokenizer
from src.data.tokenization.huggingface import HuggingFaceTokenizer
from src.data.collator import Collator
from src.data.datapipe import TranslationDataPipeline

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    config = load_config("src/configs/base_config.yaml")
    
    dataset = DatasetFactory.load_translation_dataset()
    train_data = dataset['train'].select(range(1000))
    
    print("\nRaw data sample:")
    for i in range(2):
        print(f"EN: {train_data[i]['translation']['en']}")
        print(f"RU: {train_data[i]['translation']['ru']}\n")
    
    if config['tokenizer_type'] == 'bpe':
        src_tokenizer = BPETokenizer(**config['tokenizer_params']['src'])
        tgt_tokenizer = BPETokenizer(**config['tokenizer_params']['tgt'])
        
        print("\nTraining BPE tokenizers...")
        en_texts = [x['translation']['en'] for x in train_data]
        ru_texts = [x['translation']['ru'] for x in train_data]
        src_tokenizer.train(en_texts, num_merges=500)
        tgt_tokenizer.train(ru_texts, num_merges=500)
        
        print("\nSource tokenizer vocab size:", len(src_tokenizer.token2id))
        print("Target tokenizer vocab size:", len(tgt_tokenizer.token2id))
        print("Top 10 source tokens:", list(src_tokenizer.token2id.items())[:10])
        
        # Add decode method if not available
        if not hasattr(BPETokenizer, 'decode'):
            def decode_tokens(tokenizer, token_ids):
                # Simple implementation to join tokens
                tokens = []
                for token_id in token_ids:
                    if token_id in [0, 2, 3]:  # Skip special tokens
                        continue
                    for token, idx in tokenizer.token2id.items():
                        if idx == token_id:
                            tokens.append(token)
                            break
                return ''.join(tokens).replace('</w>', ' ').strip()
            
            # Monkey patch the decode method
            src_tokenizer.decode = lambda ids: decode_tokens(src_tokenizer, ids)
            tgt_tokenizer.decode = lambda ids: decode_tokens(tgt_tokenizer, ids)
    else:
        src_tokenizer = HuggingFaceTokenizer(**config['tokenizer_params']['src'])
        tgt_tokenizer = HuggingFaceTokenizer(**config['tokenizer_params']['tgt'])
    
    collator = Collator(src_tokenizer, tgt_tokenizer, **config['collator_params'])
    
    def process_batch(batch):
        batch_src_ids = []
        batch_tgt_ids = []
        
        # Process each example in the batch
        for i in range(len(batch['id'])):
            en_text = batch['translation'][i]['en']
            ru_text = batch['translation'][i]['ru']
            
            batch_src_ids.append(src_tokenizer.tokenize(en_text))
            batch_tgt_ids.append(tgt_tokenizer.tokenize(ru_text))
        
        return {
            'src_ids': batch_src_ids,
            'tgt_ids': batch_tgt_ids
        }
    
    pipeline = TranslationDataPipeline(
        tokenizer_fn=process_batch,
        collator_fn=collator,
        **config['pipeline_params']
    )
    
    processed_data = pipeline(train_data)
    print("\nProcessed dataset sample:")
    
    print("First two examples:")
    for i in range(min(2, len(processed_data))):
        print(f"Example {i}:")
        example = processed_data[i]
        for key in example:
            print(f"  {key}: {example[key]}")
    
    print("\nDecoded sample:")
    sample = processed_data[0]['src_ids']
    print("Source IDs:", sample)
    print("Decoded:", src_tokenizer.decode(sample))
    
    sample = processed_data[0]['tgt_ids']
    print("Target IDs:", sample)
    print("Decoded:", tgt_tokenizer.decode(sample))

if __name__ == "__main__":
    main()