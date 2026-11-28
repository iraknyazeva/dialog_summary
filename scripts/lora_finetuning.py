import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unsloth  
from unsloth import FastLanguageModel

import yaml
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from src.utils.seed import set_seed

def format_example(example, tokenizer):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that summarizes dialogues. "
                "Write a concise summary in 1-2 sentences from an observer's perspective. "
                "Use formal language. Do not continue the dialogue. Write only the summary."
            )
        },
        {"role": "user", "content": f"Dialogue:\n{example['dialogue']}"},
        {"role": "assistant", "content": example['summary']}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {'text': text}

def main():
    with open('configs/data.yaml') as f:
      data_cfg = yaml.safe_load(f)

    with open('configs/lora_finetuning.yaml') as f:
      lora_cfg = yaml.safe_load(f)

    set_seed(data_cfg['seed'])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_cfg['model_name'],
        max_seq_length=lora_cfg['max_seq_length'],
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg['target_modules'],
        lora_alpha=lora_cfg['lora_alpha'],
        lora_dropout=lora_cfg['lora_dropout'],
        bias='none',
        use_gradient_checkpointing='unsloth',
        random_state=lora_cfg['seed'],
    )

    # train
    train_df = pd.read_csv(data_cfg['train_path'])
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(
        lambda x: format_example(x, tokenizer),
        remove_columns=train_dataset.column_names
    )

    # validation
    val_df = pd.read_csv(data_cfg['val_path'])
    val_dataset = Dataset.from_pandas(val_df)
    val_dataset = val_dataset.map(
        lambda x: format_example(x, tokenizer),
        remove_columns=val_dataset.column_names
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            output_dir=lora_cfg['output_dir'],
            num_train_epochs=lora_cfg['num_train_epochs'],
            per_device_train_batch_size=lora_cfg['per_device_train_batch_size'],
            gradient_accumulation_steps=lora_cfg['gradient_accumulation_steps'],
            learning_rate=lora_cfg['learning_rate'],
            logging_steps=lora_cfg['logging_steps'],
            eval_strategy='steps',
            eval_steps=lora_cfg['eval_steps'],
            save_strategy='epoch',
            logging_dir=lora_cfg.get('logging_dir', f"{lora_cfg['output_dir']}/logs"),
            report_to=lora_cfg['report_to'],
            seed=lora_cfg['seed'],
            max_seq_length=lora_cfg['max_seq_length'],
            dataset_text_field='text',
        ),
    )

    trainer.train()
    model.save_pretrained(lora_cfg['output_dir'])
    tokenizer.save_pretrained(lora_cfg['output_dir'])

if __name__ == "__main__":
    main()