"""Prepare the requested HPC SFT batch without changing existing releases."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import random


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(root, base4, base7, protocol_file, pilot=False):
    import importlib.util
    spec = importlib.util.spec_from_file_location('protocol', protocol_file)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    template = json.loads((root/'data/alfworld.yaml').read_text())
    for domain in ('alfworld', 'webshop'):
        cfg = dict(template, model_name_or_path=str(base7), template='qwen',
                   dataset_dir=str(root/'data'/f'{domain}-data'),
                   output_dir=str(root/'results'/f'{domain}7b-train'))
        cfg.pop('enable_thinking', None)
        save(root/f'{domain}7b.yaml', cfg)
    source = root/'code/goldpaths-all.zip'
    with zipfile.ZipFile(source) as z:
        gold = json.loads(z.read(z.namelist()[0]))
    rows = {'train': 0, 'validation': 0}; ids = set(); counts = {}; duplicates = 0
    data = root/('data/scienceworld-pilot-data' if pilot else 'data/scienceworld-data'); data.mkdir(parents=True, exist_ok=True)
    handles = {s:(data/f'{s}.json').open('w',encoding='utf-8') for s in rows}
    for h in handles.values(): h.write('[')
    for task in gold.values():
        sequences = task['goldActionSequences']
        if pilot:
            selected=[]
            for fold,limit in [('train',4),('dev',1)]:
                candidates=[s for s in sequences if s['fold']==fold]
                random.Random(0).shuffle(candidates)
                selected.extend(candidates[:limit])
            sequences=selected
        for sequence in sequences:
            # Official dev is validation only. Never read test path content into examples.
            fold = sequence['fold']
            if fold not in ('train', 'dev'): continue
            identity = (task['taskName'], sequence['variationIdx'])
            if identity in ids:
                duplicates += 1; continue
            ids.add(identity)
            steps = sequence['path']
            if not steps or steps[0]['action'] != 'look around':
                raise ValueError('An initial observation anchor is required')
            split = 'train' if fold == 'train' else 'validation'
            current = steps[0]['observation']; history = []
            # First look supplies the initial observation, not future evidence.
            for step in steps[1:]:
                messages = module.history_messages(module.PROMPT,
                    f"Task: {sequence['taskDescription']}", history, current, len(history))
                # ShareGPT requires alternating roles. Join only the adjacent
                # task and first observation; preserve every word and chronology.
                if messages[1]['role']==messages[2]['role']=='user':
                    messages[1]['content'] += '\n\n' + messages[2]['content']
                    del messages[2]
                action = step['action']
                messages.append({'role':'assistant','content':f'<think></think><action>{action}</action>'})
                if rows[split]: handles[split].write(',\n')
                json.dump({'messages':messages},handles[split],ensure_ascii=False,separators=(',',':'))
                rows[split] += 1
                history.append((current, action)); current = step['observation']
            counts.setdefault(task['taskName'], {'train':0,'dev':0})[fold] += 1
    for h in handles.values(): h.write(']'); h.close()
    info = {s:{'file_name':f'{s}.json','formatting':'sharegpt','columns':{'messages':'messages'},
        'tags':{'role_tag':'role','content_tag':'content','user_tag':'user','assistant_tag':'assistant','system_tag':'system'}} for s in rows}
    save(data/'dataset_info.json', info)
    save(data/'manifest.json', {'source':'allenai/ScienceWorld/goldpaths/goldpaths-all.zip',
        'revision':'e8216d6044e8e39be9fcb185e3b2dfb602584b52', 'pilot':pilot,
        'sampling':'seed 0, max 4 train + 1 dev trajectories per official task type' if pilot else 'complete official train/dev',
        'instances':[{'task':t,'variation':v} for t,v in sorted(ids)],
        'sha256':hashlib.sha256(source.read_bytes()).hexdigest(), 'counts':counts,
        'samples':rows, 'duplicate_variations_skipped':duplicates,
        'supervision':'Official actions only; empty think wrapper, no invented reasoning.',
        'observations':'Previous action response only; freelook/inventory excluded; full executed-action history.',
        'role_conversion':'Adjacent initial task and observation user messages joined with two newlines for ShareGPT; native evaluation unchanged.',
        'budget':'Official demonstrations may exceed 100 decisions; SFT task text omits a synthetic runtime budget. Evaluation retains its 100-decision limit.',
        'limitations':'Official archive coverage/version must be checked against native simulator; not full benchmark success.'})
    name='scienceworld4b-pilot' if pilot else 'scienceworld4b'
    cfg = dict(template, model_name_or_path=str(base4), dataset_dir=str(data),
               output_dir=str(root/'results'/f'{name}-train'))
    save(root/f'{name}.yaml', cfg)
    print(json.dumps({'science_samples':rows,'task_types':len(counts)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('root','base4','base7','protocol-file'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--pilot',action='store_true')
    a = p.parse_args(); prepare(a.root,a.base4,a.base7,a.protocol_file,a.pilot)
