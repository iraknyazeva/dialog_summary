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
from src.evaluate.compute_metrics import compute_metrics

def extract_first_last(dialogue):
    sentences = re.split(r'(?<=[.!?])\s+', dialogue.strip())
    sentences = [s.strip() for s in sentences]

    return f"{sentences[0]} {sentences[-1]}"

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