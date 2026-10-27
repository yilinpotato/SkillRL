"""Merge a completed SFT adapter on CPU and verify the exported model tensors."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--name', required=True)
    args = p.parse_args()
    completed = json.loads((args.run_dir / 'completed.json').read_text())
    if args.output.exists():
        raise ValueError('Never overwrite a model version')
    torch.set_num_threads(4)
    base = AutoModelForCausalLM.from_pretrained(completed['base_model'],
        torch_dtype=torch.bfloat16, local_files_only=True)
    model = PeftModel.from_pretrained(base, args.run_dir / 'adapter').merge_and_unload()
    model.save_pretrained(args.output, max_shard_size='1GB', safe_serialization=True)
    AutoTokenizer.from_pretrained(args.run_dir / 'adapter').save_pretrained(args.output)
    state = model.state_dict()
    tensors = 0
    for shard in args.output.glob('*.safetensors'):
        restored = load_file(str(shard), device='cpu')
        for key, value in restored.items():
            if not torch.equal(value, state[key]):
                raise ValueError('Exported tensor differs: ' + key)
            tensors += 1
    provenance = {k: completed[k] for k in ('metrics', 'dataset_manifest', 'protocol')}
    provenance.update(model_id=args.name, base_model='Qwen/Qwen3-4B-Thinking-2507',
        stage='SFT only; no RL; original SkillRL instruction/output',
        training={'epochs':3,'learning_rate':1e-5,'lora_r':16,'lora_alpha':32,
                  'target_modules':'all-linear','microbatch':1,'gradient_accumulation':16,'seed':0},
        validation='Exported tensors equal merged tensors; no benchmark score claimed')
    (args.output / 'sft_provenance.json').write_text(json.dumps(provenance, indent=2))
    (args.output / 'README.md').write_text(
        f'---\nlicense: apache-2.0\ntask: text-generation\n---\n\n# {args.name}\n\n'
        'BF16 merged Qwen3-4B-Thinking-2507 model, LoRA SFT on the pinned official '
        'SkillRL dataset. No RL. See sft_provenance.json for exact data and training metrics.\n\n'
        'Source instructions and targets, including retrieved skills, were retained. '
        'The source includes synthetic/placeholder observations and, for ALFWorld, '
        'synthetic done steps. These are not verified environment replays. '
        'Task-group validation loss is not benchmark success rate.\n\n'
        'Output interface: <think>...</think><action>...</action>. '
        'No benchmark environment or scoring changes were made.\n')
    manifest = {'model_id':args.name, 'base_model':provenance['base_model'],
        'tensor_count':tensors, 'tensor_values_identical_to_merged_model':True,
        'files':[{'name':f.name,'bytes':f.stat().st_size,'sha256':sha256(f),'kind':'model'}
                 for f in sorted(args.output.iterdir()) if f.is_file()]}
    (args.output / 'release-manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.output / 'SHA256SUMS').write_text(''.join(f"{x['sha256']}  {x['name']}\n" for x in manifest['files']))
    print(json.dumps({'model_id':args.name,'verified_tensors':tensors}),flush=True)


if __name__ == '__main__':
    main()
