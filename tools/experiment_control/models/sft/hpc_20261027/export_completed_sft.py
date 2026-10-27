"""CPU-export a completed batch adapter; publish provenance after tensor verification."""
import argparse
import json
from pathlib import Path
from restore_search_release import sha

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--key', choices=['alfworld7b', 'webshop7b', 'scienceworld4b-pilot'], required=True)
p.add_argument('--model-id', required=True)
a = p.parse_args()
cfg = json.loads((a.root/(a.key+'.yaml')).read_text())
run = Path(cfg['output_dir'])
state = json.loads((run/'trainer_state.json').read_text())
if state['epoch'] < 2.999:
    raise ValueError('SFT has not completed three epochs')
out = a.root/'releases'/a.model_id
if out.exists():
    raise ValueError('Existing export must be reviewed, never overwritten')
print('Loading completed adapter for CPU export', flush=True)
import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(2)
base = AutoModelForCausalLM.from_pretrained(cfg['model_name_or_path'],
    torch_dtype=torch.bfloat16, device_map='cpu', local_files_only=True)
model = PeftModel.from_pretrained(base, run).merge_and_unload()
model.save_pretrained(out, max_shard_size='1GB', safe_serialization=True)
AutoTokenizer.from_pretrained(cfg['model_name_or_path'], local_files_only=True).save_pretrained(out)
tensors = model.state_dict()
count = 0
for f in out.glob('*.safetensors'):
    for name, tensor in load_file(str(f), device='cpu').items():
        if not tensor.equal(tensors[name]):
            raise ValueError('Saved tensor mismatch: '+name)
        count += 1
portable = {k:v for k,v in cfg.items() if k not in ('model_name_or_path','dataset_dir','output_dir')}
source = Path(cfg['dataset_dir'])/'manifest.json'
provenance = {'model_id': a.model_id, 'base_model': Path(cfg['model_name_or_path']).name,
    'framework': 'LLaMA-Factory 0.9.3', 'training': portable, 'epochs': state['epoch'],
    'optimizer_steps': state['global_step'], 'adapter_sha256': sha(run/'adapter_model.safetensors'),
    'dataset_manifest': json.loads(source.read_text()) if source.exists() else None,
    'metrics': json.loads((run/'all_results.json').read_text()),
    'stage': 'SFT only; not benchmark evaluation', 'verified_tensors': count}
(out/'sft_provenance.json').write_text(json.dumps(provenance, indent=2))
manifest = {'model_id': a.model_id, 'verified_tensors': count,
    'files': [{'name': f.name, 'bytes': f.stat().st_size, 'sha256': sha(f)}
              for f in sorted(out.iterdir()) if f.is_file()]}
(out/'release-manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps({'status': 'verified', 'model_id': a.model_id, 'tensor_count': count}), flush=True)
