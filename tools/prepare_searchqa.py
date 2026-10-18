"""Materialize MRQA SearchQA with a deterministic held-out learning split."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='data/searchqa')
    parser.add_argument('--dataset', default='lucadiliello/searchqa')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--limit', type=int, default=0, help='Per-split cap, 0 exports all examples')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision = HfApi().dataset_info(args.dataset, revision=args.revision).sha
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use an empty data output directory')
    counts = {}
    for source_split in ('train', 'validation'):
        dataset = load_dataset(args.dataset, split=source_split, revision=revision, streaming=False)
        destinations = ('train', 'validation') if source_split == 'train' else ('test',)
        handles = {split: (output / f'{split}.jsonl').open('w', encoding='utf-8') for split in destinations}
        counts.update({split: 0 for split in destinations})
        iterator = iter(dataset)
        try:
            for row in iterator:
                answers = row.get('answers', row.get('answer'))
                if isinstance(answers, str):
                    answers = [answers]
                identity = row.get('key', row.get('id'))
                if identity is None or not answers:
                    raise ValueError('Dataset requires key/id and answer/answers fields')
                bucket = int(hashlib.sha256(str(identity).encode()).hexdigest()[:8], 16) % 100
                split = ('validation' if bucket < 5 else 'train') if source_split == 'train' else 'test'
                if args.limit and counts[split] >= args.limit:
                    continue
                handles[split].write(json.dumps({'id': str(identity), 'question': row['question'],
                                         'answers': answers}, ensure_ascii=False) + '\n')
                counts[split] += 1
                if args.limit and all(counts[s] >= args.limit for s in destinations):
                    break
        finally:
            if hasattr(iterator, 'close'):
                iterator.close()
            for handle in handles.values():
                handle.close()
    (output / 'source.json').write_text(json.dumps({
        'dataset': args.dataset, 'revision': revision, 'counts': counts,
        'protocol': 'searchqa_live', 'stored_search_snippets_used': False,
        'split_policy': 'Source train: sha256(key) first 8 hex modulo 100 < 5 becomes validation; '
                        'remaining train stays train. Source validation becomes held-out test. '
                        'These are project splits, not original SearchQA official splits.',
    }, indent=2), encoding='utf-8')
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
