"""CPU merge of a completed LLaMA-Factory adapter; keep the base inference template."""
import argparse
import json
import sys
from pathlib import Path


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from safetensors.torch import load_file
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
    from modelscope_transfer import sha256
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--environment',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();key=args.environment
    config=json.loads((args.root/(key+'.yaml')).read_text());run=args.root/(key+'-train')
    state=json.loads((run/'trainer_state.json').read_text())
    if state['epoch']<2.999: raise ValueError('Training has not completed three epochs')
    if args.output.exists(): raise ValueError('Never replace an existing release')
    torch.set_num_threads(4)
    base=AutoModelForCausalLM.from_pretrained(config['model_name_or_path'],torch_dtype=torch.bfloat16,local_files_only=True)
    model=PeftModel.from_pretrained(base,run).merge_and_unload()
    model.save_pretrained(args.output,max_shard_size='1GB',safe_serialization=True)
    AutoTokenizer.from_pretrained(config['model_name_or_path']).save_pretrained(args.output)
    tensors=model.state_dict();count=0
    for file in args.output.glob('*.safetensors'):
        for name,value in load_file(str(file),device='cpu').items():
            assert value.equal(tensors[name]),name
            count+=1
    portable={k:v for k,v in config.items() if k not in ('model_name_or_path','dataset_dir','output_dir')}
    provenance={'model_id':args.output.name,'base_model':'Qwen/Qwen3-4B-Thinking-2507',
        'framework':'LLaMA-Factory 0.9.3','training':portable,'optimizer_steps':state['global_step'],
        'dataset_manifest':json.loads((args.root/(key+'-data')/'manifest.json').read_text()),
        'metrics':json.loads((run/'all_results.json').read_text()),
        'epoch_evaluations':[x for x in state['log_history'] if 'eval_loss' in x],
        'protocol':'live-search-v4-search-answer' if key=='search' else 'existing-think-action',
        'training_template':'LLaMA-Factory qwen3; final assistant only; original target reasoning retained',
        'inference_tokenizer':'Exact base tokenizer and Thinking-2507 chat template',
        'stage':'SFT only; not an exact reproduction of official SkillRL 7B SFT'}
    name='searchqa_sft_provenance.json' if key=='search' else 'sft_provenance.json'
    (args.output/name).write_text(json.dumps(provenance,indent=2))
    (args.output/'README.md').write_text(f'---\nlicense: apache-2.0\ntask: text-generation\n---\n\n# {args.output.name}\n\n'
        'LLaMA-Factory 0.9.3 SFT, learning rate 1e-4, 3 epochs, LoRA r16/alpha32/all-linear. '
        'Base: Qwen3-4B-Thinking-2507. Exact v1 data split reused. BF16 merged weights; no RL.\n\n'
        'Search uses the existing search/answer adaptation without SkillBank. ALFWorld/WebShop '
        'retain original SkillRL instructions, targets and teacher skill context. '
        'Source data includes synthetic demonstrations; validation loss is not benchmark success. '
        'No claim of reproducing SkillRL official 7B SFT. See provenance JSON.\n')
    manifest={'model_id':args.output.name,'base_model':provenance['base_model'],
        'tensor_count':count,'tensor_values_identical_to_merged_model':True,
        'files':[{'name':f.name,'bytes':f.stat().st_size,'sha256':sha256(f),'kind':'model'}
                 for f in sorted(args.output.iterdir()) if f.is_file()]}
    (args.output/'release-manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.output/'SHA256SUMS').write_text(''.join(f"{x['sha256']}  {x['name']}\n" for x in manifest['files']))
    print(json.dumps({'model_id':args.output.name,'verified_tensors':count}),flush=True)


if __name__=='__main__': main()
