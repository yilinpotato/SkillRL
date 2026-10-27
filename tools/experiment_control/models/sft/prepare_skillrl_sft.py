"""Prepare pinned, unmodified SkillRL ALFWorld/WebShop targets for response-only SFT.
Source text is retained, including teacher skill context and synthetic observations.
"""
import argparse, hashlib, json, re
from pathlib import Path
import pandas as pd

REVISION = 'bd3996c4e863ac59e6e4ab35549cf3741faf5e4f'
HASHES = {'alfworld':'dfbbf265e19ac8087a54ec474727fcb400483a9a243eea6e977a02ae6ca85b94',
          'webshop':'2c4f045b18a7ffabf7779f8e0e416913debae6427cb70a323c79e30f34d2d051'}

def convert(row, index):
    instruction, output = row['instruction'], row['output']
    task = instruction.split('Your task is to: ',1)[1].split('\n',1)[0].strip()
    if not re.fullmatch(r'\s*<think>.*?</think>\s*<action>.*?</action>\s*',output,re.S):
        raise ValueError(f'Malformed target at row {index}; inspect rather than silently drop')
    task_key = ' '.join(task.lower().split())
    return {'source_row':int(index), 'task':task,
        'task_id':hashlib.sha256(task_key.encode()).hexdigest(),
        'prompt_messages':[{'role':'user','content':instruction}], 'response':output}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--environment',choices=HASHES,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); digest=hashlib.sha256(args.source.read_bytes()).hexdigest()
    if digest!=HASHES[args.environment]: raise ValueError('Source checksum mismatch')
    if args.output.exists(): raise ValueError('Use a new dataset directory')
    rows=[convert(row,i) for i,row in pd.read_parquet(args.source).iterrows()]
    tasks=sorted({r['task_id'] for r in rows}); val=set(tasks[:max(1,round(len(tasks)*.05))])
    args.output.mkdir(parents=True); splits={}
    for split in ('train','validation'):
        subset=[r for r in rows if (r['task_id'] in val)==(split=='validation')]
        (args.output/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subset))
        splits[split]={'samples':len(subset),'tasks':len({r['task_id'] for r in subset})}
    audit={'synthetic_done_targets':0,'target_not_in_declared_actions':0,'generic_product_observation_rows':0}
    for r in rows:
        action=re.search(r'<action>(.*?)</action>',r['response'],re.S).group(1).strip()
        text=r['prompt_messages'][0]['content']; current=text.rsplit('Your admissible actions of the current situation are: [',1)[-1].split('].',1)[0]
        pool=[x.strip().lower() for x in current.split(',')]
        audit['synthetic_done_targets']+=action.lower()=='done'
        audit['target_not_in_declared_actions']+=action.lower() not in pool
        audit['generic_product_observation_rows']+='Product A - Matches your search' in text
    manifest={'environment':args.environment,'source':'Jianwen/SkillRL-SFT-Data/'+args.environment,
        'revision':REVISION,'source_sha256':digest,'splits':splits,'audit':audit,
        'transformation':'Original instruction/output retained verbatim; user chat wrapper only',
        'split':'First 5 percent of sorted normalized-task SHA256 groups used for validation',
        'skill_context':'Preserved original SkillRL retrieved teacher context',
        'benchmark_eval_overlap':'Not established: source has no environment instance IDs; internal validation is not benchmark evaluation',
        'protocol':'existing-think-action; benchmark interfaces unchanged',
        'quality_note':'Official data includes synthetic/placeholder observations and synthetic ALFWorld done steps; not verified environment replays'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
