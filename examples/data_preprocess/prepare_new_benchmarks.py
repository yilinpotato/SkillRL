"""Convert explicit disjoint manifests to native VERL environment rows."""
import argparse
import json
from pathlib import Path
import pandas as pd
from agent_system.environments.benchmark_protocol import load_examples, check_disjoint


def prepare(kind: str, paths: dict, output: Path) -> None:
    if output.exists():
        raise ValueError('Use a new output directory')
    if kind == 'searchqa_live':
        splits = {split: load_examples(path) for split, path in paths.items()}
        for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
            check_disjoint(splits[a], splits[b])
    else:
        # Supply the same frozen manifest used by CoSkill, with official variation IDs.
        manifest = json.loads(paths['train'].read_text())
        splits = {split: manifest[split] for split in ('train', 'dev', 'test')}
        seen = set()
        for rows in splits.values():
            if not rows:
                raise ValueError('Empty ScienceWorld split')
            for row in rows:
                identity = (row['task'], int(row['variation']))
                if identity in seen:
                    raise ValueError('Duplicate or overlapping ScienceWorld instance')
                seen.add(identity)
    output.mkdir(parents=True)
    for split, rows in splits.items():
        records = []
        for i, row in enumerate(rows):
            question = row.get('question', row.get('task'))
            records.append(dict(data_source=kind, prompt=[{'role': 'user', 'content': question}],
                                ability='agent', reward_model={'style': 'rule', 'ground_truth': ''},
                                extra_info={'index': i, 'split': split}, env_kwargs=dict(row, split=split)))
        pd.DataFrame(records).to_parquet(output / (split + '.parquet'), index=False)
    (output / 'manifest.json').write_text(json.dumps({'benchmark': kind, 'sizes': {k: len(v) for k, v in splits.items()},
                                                     'evaluation_answers': 'env_kwargs only; never in model messages'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', choices=['searchqa_live', 'scienceworld'], required=True)
    p.add_argument('--train', type=Path, required=True, help='SearchQA train JSONL, or complete ScienceWorld manifest JSON')
    p.add_argument('--dev', type=Path)
    p.add_argument('--test', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.benchmark == 'searchqa_live' and (args.dev is None or args.test is None):
        p.error('SearchQA requires explicit train/dev/test files')
    prepare(args.benchmark, {'train': args.train, 'dev': args.dev, 'test': args.test}, args.output)
