"""Response-only LoRA SFT with the Search SFT hyperparameters; environment-specific source data."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


def encode(row, tokenizer, max_length):
    prompt = tokenizer.apply_chat_template(row['prompt_messages'], tokenize=False,
                                          add_generation_prompt=True)
    prefix = tokenizer(prompt, add_special_tokens=False)['input_ids']
    target = row['response']
    if prompt.rstrip().endswith('<think>') and target.startswith('<think>'):
        target = target[len('<think>'):]
    response = tokenizer(target + tokenizer.eos_token, add_special_tokens=False)['input_ids']
    if len(prefix) + len(response) > max_length:
        raise ValueError(f'Sample {row["source_row"]} exceeds max_length; no silent truncation')
    return {'input_ids': prefix + response, 'attention_mask': [1] * (len(prefix) + len(response)),
            'labels': [-100] * len(prefix) + response}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--max-length', type=int, default=16384)
    p.add_argument('--epochs', type=float, default=3)
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    output = Path(args.output_dir)
    if not args.check_only and output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    data = {}
    for split in ('train', 'validation'):
        rows = [json.loads(line) for line in (Path(args.data_dir) / f'{split}.jsonl').read_text().split('\n') if line.strip()]
        data[split] = [encode(row, tokenizer, args.max_length) for row in rows]
    print(json.dumps({key: {'samples': len(rows), 'max_tokens': max(len(x['input_ids']) for x in rows),
                           'supervised_tokens': sum(sum(t != -100 for t in x['labels']) for x in rows)}
                      for key, rows in data.items()}), flush=True)
    if args.check_only:
        return
    from peft import LoraConfig, get_peft_model
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires an explicitly allocated GPU')
    set_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                               attn_implementation='sdpa')
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules='all-linear',
                                           task_type='CAUSAL_LM'))
    def collate(rows):
        length = max(len(x['input_ids']) for x in rows)
        return {key: torch.tensor([x[key] + [pad] * (length - len(x[key])) for x in rows])
                for key, pad in [('input_ids', tokenizer.pad_token_id), ('attention_mask', 0), ('labels', -100)]}
    training = TrainingArguments(output_dir=str(output), num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=16, learning_rate=1e-5, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        dataloader_num_workers=0, logging_steps=1, save_total_limit=1, save_strategy='epoch', eval_strategy='epoch', report_to=[], seed=0)
    trainer = Trainer(model=model, args=training, train_dataset=data['train'],
                      eval_dataset=data['validation'], data_collator=collate)
    result = trainer.train()
    trainer.save_model(str(output / 'adapter'))
    tokenizer.save_pretrained(output / 'adapter')
    (output / 'completed.json').write_text(json.dumps({'metrics': result.metrics,
        'dataset_manifest': json.loads((Path(args.data_dir) / 'manifest.json').read_text()),
        'training_args': vars(args),
        'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
        'base_model': args.model, 'protocol': 'existing-think-action',
        'artifact': 'LoRA adapter; merge with its exact base before evaluation or RL'}, indent=2))


if __name__ == '__main__':
    main()
