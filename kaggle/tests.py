import torch
from transformers import AutoTokenizer
from .architecture import MultiHeadAttention, Transformer
from .data_utils import translate_sentence

def test_tokenization(tokenizer, test_sentence="Hello, how are you?"):
    print("="*50)
    print("1. Testing tokenization and special tokens")
    
    print(f"BOS: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
    print(f"EOS: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    
    processed_text = f"{tokenizer.bos_token} {test_sentence} {tokenizer.eos_token}"
    
    encoded = tokenizer(
        processed_text,
        padding='max_length',
        truncation=True,
        max_length=20, 
        return_tensors='pt',
        add_special_tokens=False
    )
    
    token_ids = encoded['input_ids'][0].tolist()
    decoded = tokenizer.decode(token_ids)
    
    print("\nTokenization example:")
    print(f"Tokens: {token_ids}")
    print(f"Decoded: {decoded}")
    
    try:
        eos_pos = token_ids.index(tokenizer.eos_token_id)
        pad_start = token_ids.index(tokenizer.pad_token_id)
        assert eos_pos < pad_start, "EOS should be before padding"
    except ValueError:
        assert False, "EOS token not found"
    
    assert token_ids[0] == tokenizer.bos_token_id
    print("\nTokenization is working correctly")

def test_masking():
    print("\n" + "="*50)
    print("2. Testing mask generation")
    
    batch_size = 2
    seq_len = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    src = torch.ones(batch_size, seq_len, device=device)
    tgt = torch.ones(batch_size, seq_len, device=device)
    
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    tgt_mask = causal_mask.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    print("\nSource mask (example):")
    print(src_mask[0, 0, 0].cpu().numpy()) 
    
    print("\nTarget mask (example):")
    print(tgt_mask[0, 0].cpu().numpy())
    
    # Check dimensions
    assert src_mask.shape == (batch_size, 1, 1, seq_len), f"Expected (2,1,1,5), got {src_mask.shape}"
    assert tgt_mask.shape == (batch_size, 1, seq_len, seq_len), f"Expected (2,1,5,5), got {tgt_mask.shape}"
    
    print("\nMasks are created correctly")

def test_model_forward_pass(model, tokenizer, device):
    print("\n" + "="*50)
    print("3. Testing model forward pass")
    
    test_sentence = "Test input"
    inputs = tokenizer(
        test_sentence,
        return_tensors="pt",
        padding='max_length',
        max_length=128,
        truncation=True
    ).to(device)
    
    src = inputs['input_ids']
    tgt = torch.cat([
        torch.tensor([[tokenizer.bos_token_id]], device=device),
        src[:, :-1]
    ], dim=1)

    src_mask = inputs['attention_mask'].to(torch.bool).unsqueeze(1).unsqueeze(2)
    seq_len = tgt.size(1)
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).to(torch.bool)
    tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        output = model(src, tgt, src_mask, tgt_mask)
    
    print("\nModel output shape:", output.shape)
    print("Minimum value:", output.min().item())
    print("Maximum value:", output.max().item())
    
    assert not torch.isnan(output).any()
    expected_vocab_size = len(tokenizer)  # Use the actual vocabulary size
    assert output.shape == (1, 128, expected_vocab_size), (
        f"Expected shape: (1, 128, {expected_vocab_size}), "
        f"got: {output.shape}"
    )
    print("\nForward pass is working correctly")

def test_dynamic_masking():
    print("Testing dynamic masks for different sizes")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_heads = 8
    d_model = 512
    
    test_cases = [
        {'batch_size': 1, 'seq_len': 7},
        {'batch_size': 3, 'seq_len': 15},
        {'batch_size': 5, 'seq_len': 3}
    ]
    
    for case in test_cases:
        print(f"\nTest for batch={case['batch_size']} seq_len={case['seq_len']}")
        
        src = torch.randint(0, 100, (case['batch_size'], case['seq_len'])).to(device)
        tgt = torch.randint(0, 100, (case['batch_size'], case['seq_len'])).to(device)
        
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = torch.tril(torch.ones(case['seq_len'], case['seq_len'])).to(device)
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
        
        mha = MultiHeadAttention(d_model, num_heads).to(device)
        q = k = v = torch.randn(case['batch_size'], case['seq_len'], d_model).to(device)
        
        try:
            output = mha(q, k, v, tgt_mask)
            assert output.shape == (case['batch_size'], case['seq_len'], d_model)
            print("Success")
        except Exception as e:
            print(f"Error: {str(e)}")
            raise

def test_multi_head_compatibility():
    print("\nTesting mask compatibility with Multi-Head Attention")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d_model = 512
    num_heads_list = [4, 8, 16]
    seq_lens = [10, 15, 20]
    
    for num_heads in num_heads_list:
        for seq_len in seq_lens:
            print(f"\nHeads: {num_heads}, Seq_len: {seq_len}")
            
            mask = torch.tril(torch.ones(seq_len, seq_len)).to(device)
            
            mha = MultiHeadAttention(d_model, num_heads).to(device)
            q = torch.randn(2, seq_len, d_model).to(device)
            
            try:
                output = mha(q, q, q, mask)
                assert output.shape == q.shape
                print("Compatibility confirmed")
            except Exception as e:
                print(f"Incompatibility: {str(e)}")
                raise

def test_translation_function(model, tokenizer, device):
    print("\n" + "="*50)
    print("4. Testing translation function")
    
    test_sentences = [
        "Hello world",
        "How are you?",
        "Test sentence",
        "Machine learning"
    ]
    
    for sentence in test_sentences:
        print("\n" + "-"*50)
        print(f"Translation: '{sentence}'")
        
        try:
            translated = translate_sentence(
                model=model,
                tokenizer=tokenizer,
                sentence=sentence,
                device=device,
                max_length=64  # Reduced length for tests
            )
            print(f"Result: {translated}")
        except Exception as e:
            print(f"Error: {str(e)}")
            raise
    
    print("\nTranslation function is working correctly")

def test_model_with_various_inputs():
    print("\nTesting the model with various input sizes")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
        tokenizer.add_special_tokens({
            'bos_token': '<s>', 'eos_token': '</s>', 
            'pad_token': '<pad>', 'unk_token': '<unk>'
        })
    except Exception as e:
        print(f"Tokenizer initialization error: {e}")
        raise
    
    d_model = 512
    num_heads = 8
    num_layers = 6
    
    model = Transformer(
        src_vocab_size=len(tokenizer),
        tgt_vocab_size=len(tokenizer),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers
    ).to(device)
    model.eval()
    
    test_cases = [
        {'batch_size': 1, 'src_len': 10, 'tgt_len': 12},
        {'batch_size': 3, 'src_len': 7, 'tgt_len': 9},
        {'batch_size': 5, 'src_len': 15, 'tgt_len': 15},
        {'batch_size': 2, 'src_len': 5, 'tgt_len': 5}
    ]
    
    for case in test_cases:
        print(f"\nTest: batch={case['batch_size']} src={case['src_len']} tgt={case['tgt_len']}")
        
        try:
            src = torch.randint(0, len(tokenizer), 
                             (case['batch_size'], case['src_len'])).to(device)
            tgt = torch.randint(0, len(tokenizer), 
                             (case['batch_size'], case['tgt_len'])).to(device)
            
            src_mask = (src != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
            tgt_mask = torch.tril(torch.ones(
                (case['batch_size'], case['tgt_len'], case['tgt_len']),
                device=device
            )).bool().unsqueeze(1)
            
            with torch.no_grad():
                output = model(src, tgt, src_mask, tgt_mask)
            
            expected_shape = (
                case['batch_size'], 
                case['tgt_len'], 
                len(tokenizer)
            )
            assert output.shape == expected_shape
            print("Working correctly")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            raise

def run_comprehensive_tests():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
    tokenizer.add_tokens(['<s>', '</s>', '<pad>', '<unk>'])
    tokenizer.bos_token = '<s>'
    tokenizer.eos_token = '</s>'
    tokenizer.pad_token = '<pad>'
    tokenizer.unk_token = '<unk>'
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids('<s>')
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('</s>')
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')
    
    model = Transformer(
        src_vocab_size=len(tokenizer),
        tgt_vocab_size=len(tokenizer),
        d_model=512,
        num_heads=8,
        num_layers=6
    ).to(device)
    
    test_tokenization(tokenizer)
    test_masking()
    test_model_forward_pass(model, tokenizer, device)
    test_dynamic_masking()
    test_multi_head_compatibility()
    test_model_with_various_inputs()
    test_translation_function(model, tokenizer, device)