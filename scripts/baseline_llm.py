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

def build_messages(dialogue):
    # пример 1
    ex1_dialogue = """#Person1#: Hi, Mr. Smith. I'm Doctor Hawkins. Why are you here today?
#Person2#: I found it would be a good idea to get a check-up.
#Person1#: Yes, well, you haven't had one for 5 years. You should have one every year.
#Person2#: I know. I figure as long as there is nothing wrong, why go see the doctor?
#Person1#: Well, the best way to avoid serious illnesses is to detect them early. So try to come at least once a year for your own good.
#Person2#: Ok.
#Person1#: Let me see here. Your eyes and ears look fine. Take a deep breath, please. Do you smoke, Mr. Smith?
#Person2#: Yes.
#Person1#: Smoking is the leading cause of lung cancer and heart disease, you know. You really should quit.
#Person2#: I've tried a hundred times, but I just can't seem to kick the habit.
#Person1#: Well, we have classes and some medications that might help. I'll give you more information before you leave.
#Person2#: Ok, thanks doctor."""

    ex1_summary = "Mr. Smith's getting a check-up, and Doctor Hawkins advises him to have one every year. Hawkins'll give some information about their classes and medications to help Mr. Smith quit smoking."

    # пример 2
    ex2_dialogue = """#Person1#: Hey, man. You wanna buy some weed?
#Person2#: Some what?
#Person1#: Weed! You know? Pot, Ganja, Mary Jane some chronic!
#Person2#: Oh, umm, no thanks.
#Person1#: I also have blow if you prefer to do a few lines.
#Person2#: No, I am ok, really.
#Person1#: Come on man! I even got dope and acid! Try some!
#Person2#: Do you really have all of these drugs? Where do you get them from?
#Person1#: I got my connections! Just tell me what you want and I can get it!
#Person2#: No, I am really ok. I am just going to go home.
#Person1#: Well, if you change your mind, come find me.
#Person2#: Ok, thanks."""

    ex2_summary = "#Person1# offers some drugs to #Person2#, but #Person2# refuses them."

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
            "content": f"Dialogue:\n{ex1_dialogue}"
        },
        {
            "role": "assistant",
            "content": ex1_summary
        },
        {
            "role": "user",
            "content": f"Dialogue:\n{ex2_dialogue}"
        },
        {
            "role": "assistant",
            "content": ex2_summary
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
    
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")
    sampling_params = SamplingParams(temperature=llm_cfg['temperature'], max_tokens=llm_cfg['max_tokens'], seed = llm_cfg["seed"])
    
    outputs = llm.chat(messages_list, sampling_params)
    predictions = [out.outputs[0].text.strip() for out in outputs]
    
    metrics = compute_metrics(predictions, references)
    
    with open('metrics.txt', 'w', encoding='utf-8') as f:
        for k, v in metrics.items():
            line = f'{k}: {v}'
            print(line, flush=True)
            f.write(line + '\n')

    for i in range(min(5, len(predictions))):
        print(f"Dialogue:\n{dialogues[i]}")
        print(f"\nReference:\n{references[i]}")
        print(f"\nGenerated summary:\n{predictions[i]}")


if __name__ == "__main__":
    main()