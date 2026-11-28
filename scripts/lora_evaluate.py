import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from src.utils.seed import set_seed
from src.evaluate.compute_metrics import compute_metrics


def build_messages(dialogue):
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that summarizes dialogues. "
                "Write a concise summary in 1-2 sentences from an observer's perspective. "
                "Use formal language. Do not continue the dialogue. Write only the summary."
            )
        },
        {
            "role": "user",
            "content": f"Dialogue:\n{dialogue}"
        }
    ]


def main():
    with open('configs/data.yaml') as f:
        data_cfg = yaml.safe_load(f)
    with open('configs/baseline_llm.yaml') as f:
        llm_cfg = yaml.safe_load(f)

    set_seed(data_cfg["seed"])

    df = pd.read_csv(data_cfg["test_path"]).head(llm_cfg['num_examples'])
    dialogues = df[data_cfg["text_column"]].tolist()
    references = df[data_cfg["summary_column"]].tolist()

    messages_list = [build_messages(d) for d in dialogues]

    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        enable_lora=True,
        max_loras=1,
        max_lora_rank=16,
        dtype="auto",
    )

    sampling_params = SamplingParams(
        temperature=llm_cfg['temperature'],
        max_tokens=llm_cfg['max_tokens'],
        seed=llm_cfg["seed"],
    )

    lora_request = LoRARequest(
        lora_name="dialog_summary_lora",
        lora_int_id=1,
        lora_path="models/lora_adapter",
    )

    outputs = llm.chat(
        messages_list,
        sampling_params,
        lora_request=lora_request,
    )

    predictions = [out.outputs[0].text.strip() for out in outputs]

    metrics = compute_metrics(predictions, references)
    print(metrics)

    with open('metrics_lora.txt', 'w', encoding='utf-8') as f:
        for k, v in metrics.items():
            line = f'{k}: {v}'
            print(line)
            f.write(line + '\n')

    for i in range(min(5, len(predictions))):
        print(f"Dialogue:\n{dialogues[i]}")
        print(f"\nReference:\n{references[i]}")
        print(f"\nLoRA summary:\n{predictions[i]}\n")


if __name__ == "__main__":
    main()