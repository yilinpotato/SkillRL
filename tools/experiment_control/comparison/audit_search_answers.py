"""Read-only, per-trajectory SearchQA EM/F1 audit; never rewrites run metrics."""
import argparse
import collections
import importlib.util
import json
import re
from pathlib import Path


def prediction(trace, project):
    answer = ''
    searches = 0
    for step in trace['steps']:
        if not step.get('is_action_valid', False):
            continue
        response = step.get('protocol_response') or step.get('model_response') or step.get('raw_response') or ''
        action = project(response)['parsed_action']
        if not action:
            raise ValueError('Valid action cannot be reconstructed')
        if action.startswith('search[') and action.endswith(']'):
            searches += 1
        if action.startswith('answer[') and action.endswith(']'):
            answer = action[7:-1]
            break
    return answer, searches


def question(trace):
    if trace.get('task'):
        return trace['task']
    match = re.search(r'Question: (.*?)\nMaximum actions:', trace['steps'][0]['observation'], re.S)
    if not match:
        raise ValueError('Missing question identity')
    return match.group(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import pandas as pd
    spec = importlib.util.spec_from_file_location('pinned_protocol', args.protocol)
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    refs, dev = {}, []
    for filename in ('scheduled_train.parquet', 'fixed_dev.parquet'):
        for row in pd.read_parquet(args.run / filename).env_kwargs:
            q = row['question']
            answers = list(row['answers'])
            if q in refs and refs[q] != answers:
                raise ValueError('Conflicting answer references')
            refs[q] = answers
            if filename == 'fixed_dev.parquet':
                dev.append(q)
    groups = collections.defaultdict(list)
    sources = []
    if (args.run / 'trajectories.json').exists():
        source = args.run / 'trajectories.json'
        sources.append(str(source))
        records = json.loads(source.read_text())
        dev_ordinals = collections.Counter()
        for t in records:
            step = t['global_step']
            q = question(t)
            if q in dev:
                ordinal = dev_ordinals[step] // len(dev)
                expected = dev[dev_ordinals[step] % len(dev)]
                if q != expected:
                    raise ValueError('Evaluation order differs from fixed dev manifest')
                dev_ordinals[step] += 1
                group = f'dev/step{step}/pass{ordinal}'
            else:
                group = f'train/step{step}'
            groups[group].append(t)
    else:
        for source in sorted((args.run / 'trajectories').glob('*.jsonl')):
            sources.append(str(source))
            records = [json.loads(line) for line in source.read_text().split('\n') if line.strip()]
            split = 'dev' if all(question(t) in dev for t in records) else 'train'
            groups[f'{split}/{source.stem}'] = records
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {'source_run': str(args.run), 'sources': sources, 'unit': 'one trajectory, one submitted answer; max over reference aliases', 'historical_adamw_calls': None, 'groups': {}}
    with (args.output / 'answers.jsonl').open('w', encoding='utf-8') as out:
        for name, records in groups.items():
            totals = collections.Counter()
            for t in records:
                q = question(t)
                pred, searches = prediction(t, protocol.project_action)
                scores = protocol.score_answer(pred, refs[q])
                totals.update(scores)
                totals['searches'] += searches
                totals['missing_answer'] += not bool(pred)
                out.write(json.dumps(dict(group=name, traj_uid=t['traj_uid'], question=q, prediction=pred, searches=searches, **scores), ensure_ascii=False) + '\n')
            n = len(records)
            summary['groups'][name] = dict(trajectories=n, exact_match=totals['exact_match']/n, f1=totals['f1']/n, searches=totals['searches'], missing_answer=totals['missing_answer'], full_dev32=name.startswith('dev/') and n == len(dev), task_type_corrected='searchqa_live')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'groups': len(groups)}), flush=True)


if __name__ == '__main__':
    main()
