import numpy as np
from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from nltk import word_tokenize

def calculate_metrics(references, predictions, tokenizer):
    non_empty_indices = [i for i, pred in enumerate(predictions) if len(pred.strip()) > 0]
    valid_preds = [predictions[i] for i in non_empty_indices]
    valid_refs = [references[i] for i in non_empty_indices]

    metrics = {
        'bleu': 0.0,
        'rouge-1': {'f': 0.0},
        'rouge-2': {'f': 0.0},
        'rouge-l': {'f': 0.0},
        'meteor': 0.0,
        'meteor_precision': 0.0,
        'meteor_recall': 0.0
    }

    if not valid_preds:
        return metrics

    smooth = SmoothingFunction().method4
    refs_bleu = [[ref.strip().split()] for ref in valid_refs]
    preds_bleu = [pred.strip().split() for pred in valid_preds]
    
    if preds_bleu:
        metrics['bleu'] = corpus_bleu(refs_bleu, preds_bleu, smoothing_function=smooth)

    rouge = Rouge()
    try:
        if valid_preds:
            rouge_scores = rouge.get_scores(valid_preds, valid_refs, avg=True)
            metrics['rouge-1'] = rouge_scores['rouge-1']['f']
            metrics['rouge-2'] = rouge_scores['rouge-2']['f']
            metrics['rouge-l'] = rouge_scores['rouge-l']['f']
    except Exception as e:
        print(f"ROUGE calculation error: {str(e)}")

    meteor_scores = []
    precisions = []
    recalls = []
    
    for ref, pred in zip(valid_refs, valid_preds):
        try:
            ref_str = ref.strip()
            pred_str = pred.strip()
            
            score = single_meteor_score(ref_str, pred_str)
            meteor_scores.append(score)
            
            ref_tokens = word_tokenize(ref_str)
            pred_tokens = word_tokenize(pred_str)
            
            matches = sum(1 for word in pred_tokens if word in ref_tokens)
            precision = matches/len(pred_tokens) if len(pred_tokens) > 0 else 0.0
            recall = matches/len(ref_tokens) if len(ref_tokens) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            
        except Exception as e:
            print(f"METEOR calculation error: {str(e)}")
            meteor_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)

    if meteor_scores:
        metrics['meteor'] = np.mean(meteor_scores)
        metrics['meteor_precision'] = np.mean(precisions)
        metrics['meteor_recall'] = np.mean(recalls)

    return metrics