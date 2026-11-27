import sys
from pathlib import Path

# добавляем корень проекта в sys.path для импортов из src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml
from vllm import LLM, SamplingParams

from src.utils.seed import set_seed
from src.evaluate.compute_metrics import compute_metrics

def build_prompt(dialogue):
    return (
        "Summarize the following dialogue in 1-3 sentences. Write only the summary.\n\n"
        f"Dialogue:\n{dialogue}\n\n"
        "Summary:"
    )

def main():
    with open('configs/data.yaml') as f:
        data_cfg = yaml.safe_load(f)

    set_seed(data_cfg["seed"])
    
    df = pd.read_csv(data_cfg["test_path"]).head(100)
    dialogues = df[data_cfg["text_column"]].tolist()
    references = df[data_cfg["summary_column"]].tolist()
    
    prompts = [build_prompt(d) for d in dialogues]
    
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=80, seed = data_cfg["seed"])
    
    outputs = llm.generate(prompts, sampling_params)
    predictions = [out.outputs[0].text.strip() for out in outputs]
    
    metrics = compute_metrics(predictions, references)

    for k, v in metrics.items():
        print(f"{k}: {v}")

    for i in range(min(5, len(predictions))):
        print(f"Dialogue:\n{dialogues[i]}...")
        print(f"\nReference:\n{references[i]}")
        print(f"\nGenerated summary:\n{predictions[i]}")


if __name__ == "__main__":
    main()