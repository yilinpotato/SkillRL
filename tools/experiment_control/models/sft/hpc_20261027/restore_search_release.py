"""Reconstruct the existing Search SFT release on CPU and verify original weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for key in ('base', 'adapter', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    original = json.loads((a.adapter/'original-release-manifest.json').read_text())
    expected = [f for f in original['files'] if f['name'].endswith('.safetensors')]
    if not expected or a.output.exists():
        raise ValueError('Need original weight hashes and a fresh output directory')
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(2)
    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16,
                                              local_files_only=True, device_map='cpu')
    model = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    model.save_pretrained(a.output, max_shard_size='1GB', safe_serialization=True)
    AutoTokenizer.from_pretrained(a.base, local_files_only=True).save_pretrained(a.output)
    for entry in expected:
        f = a.output / entry['name']
        if f.stat().st_size != entry['bytes'] or sha(f) != entry['sha256']:
            raise ValueError('Reconstructed weight differs from original: ' + f.name)
    # Publish provenance only after every original weight shard matches.
    shutil.copy2(a.adapter/'searchqa_sft_provenance.json', a.output)
    report = {'model_id': original['model_id'], 'original_weight_shards_identical': True,
              'verified_weight_shards': len(expected),
              'original_manifest_sha256': sha(a.adapter/'original-release-manifest.json'),
              'adapter_sha256': sha(a.adapter/'adapter_model.safetensors'),
              'files': [{'name': f.name, 'bytes': f.stat().st_size, 'sha256': sha(f)}
                        for f in sorted(a.output.iterdir()) if f.is_file()]}
    (a.output/'release-manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'verified_weight_shards': len(expected), 'status': 'verified'}), flush=True)


if __name__ == '__main__':
    main()
