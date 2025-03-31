import torch
from torch.utils.data import DataLoader

def evaluate_model(model, val_loader, criterion, tokenizer, device, max_examples=10):
    model.eval()
    total_loss = 0
    all_sources = []
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            src_mask = (input_ids != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
            tgt_mask = torch.tril(torch.ones(
                labels.size(1)-1, 
                labels.size(1)-1, 
                device=device
            )).bool()
            
            outputs = model(
                input_ids,
                labels[:, :-1],
                src_mask,
                tgt_mask.unsqueeze(0).unsqueeze(0))
            
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                labels[:, 1:].reshape(-1))
            total_loss += loss.item()
            
            if len(all_sources) < max_examples:
                sources = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                targets = tokenizer.batch_decode(labels, skip_special_tokens=True)
                preds = [translate_sentence(model, tokenizer, src, device) for src in sources]
                all_sources.extend(sources)
                all_targets.extend(targets)
                all_predictions.extend(preds)
    
    metrics = calculate_metrics(all_targets, all_predictions, tokenizer)
    
    return (
        total_loss / len(val_loader),
        metrics,
        all_sources[:max_examples],
        all_targets[:max_examples],
        all_predictions[:max_examples]
    )