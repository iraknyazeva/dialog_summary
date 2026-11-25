import sys
from pathlib import Path

# добавляем корень проекта в sys.path для импортов из src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import json
import yaml
import pandas as pd
from rouge_score import rouge_scorer
import sacrebleu

from src.utils.seed import set_seed

def extract_first_last(dialogue):
    sentences = re.split(r'(?<=[.!?])\s+', dialogue.strip())
    sentences = [s.strip() for s in sentences]

    return f"{sentences[0]} {sentences[-1]}"

def compute_metrics(predictions, references):
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )
    
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)
    
    n = len(predictions)
    
    return {
        'BLEU': round(bleu.score, 2),
        'ROUGE-1': round(sum(rouge_scores['rouge1']) / n * 100, 2),
        'ROUGE-2': round(sum(rouge_scores['rouge2']) / n * 100, 2),
        'ROUGE-L': round(sum(rouge_scores['rougeL']) / n * 100, 2),
    }

def main():
    with open('configs/data.yaml') as f:
        data_cfg = yaml.safe_load(f)

    set_seed(data_cfg['seed'])
    
    df = pd.read_csv(data_cfg['test_path'])
    
    predictions = df[data_cfg["text_column"]].apply(extract_first_last).tolist()
    references = df[data_cfg["summary_column"]].tolist()
    
    metrics = compute_metrics(predictions, references)
    
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()