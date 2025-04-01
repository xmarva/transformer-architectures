import torch


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
                tgt_mask.unsqueeze(0).unsqueeze(0))
            
            next_token = output[:, -1].argmax(-1)
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(decoder_input[0], skip_special_tokens=True)