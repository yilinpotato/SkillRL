"""Preserve v1 samples/splits and prepare explicit LLaMA-Factory configurations."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--base',required=True)
    for key in ('search','webshop','alfworld'): p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    if args.root.exists(): raise ValueError('Use a new run directory')
    args.root.mkdir(parents=True)
    for key in ('search','webshop','alfworld'):
        source=getattr(args,key);data=args.root/(key+'-data');data.mkdir()
        info={};audit={}
        for split in ('train','validation'):
            original=source/(split+'.jsonl')
            rows=[json.loads(x) for x in original.read_text().split('\n') if x.strip()]
            converted=[]
            for row in rows:
                messages=row['prompt_messages']+[{'role':'assistant','content':row['response']}]
                converted.append({'messages':messages})
            target=data/(split+'.json')
            target.write_text(json.dumps(converted,ensure_ascii=False))
            assert all(a['messages']==b['prompt_messages']+[{'role':'assistant','content':b['response']}]
                       for a,b in zip(json.loads(target.read_text()),rows))
            audit[split]={'samples':len(rows),'source_sha256':digest(original),'converted_sha256':digest(target),
                          'message_content_identical_to_v1':True}
            info[split]={'file_name':target.name,'formatting':'sharegpt','columns':{'messages':'messages'},
                'tags':{'role_tag':'role','content_tag':'content','user_tag':'user',
                        'assistant_tag':'assistant','system_tag':'system'}}
        (data/'dataset_info.json').write_text(json.dumps(info,indent=2))
        manifest={'environment':key,'v1_source_manifest':json.loads((source/'manifest.json').read_text()),
                  'conversion':audit,'policy':'Exact v1 samples and split; no new examples or skill edits'}
        (data/'manifest.json').write_text(json.dumps(manifest,indent=2))
        config={'model_name_or_path':args.base,'stage':'sft','do_train':True,'do_eval':True,
            'finetuning_type':'lora','lora_rank':16,'lora_alpha':32,'lora_target':'all','lora_dropout':0.0,
            'dataset_dir':str(data),'dataset':'train','eval_dataset':'validation','template':'qwen3',
            'enable_thinking':True,'train_on_prompt':False,'mask_history':True,'packing':False,
            'cutoff_len':16384,'preprocessing_num_workers':1,'dataloader_num_workers':0,
            'output_dir':str(args.root/(key+'-train')),'overwrite_output_dir':False,
            'per_device_train_batch_size':1,'per_device_eval_batch_size':1,
            'gradient_accumulation_steps':16,'learning_rate':1e-4,'num_train_epochs':3.0,
            'lr_scheduler_type':'linear','warmup_ratio':0.0,'weight_decay':0.0,'max_grad_norm':1.0,
            'bf16':True,'gradient_checkpointing':True,'flash_attn':'sdpa','optim':'adamw_torch',
            'seed':0,'data_seed':0,'logging_steps':1,'save_strategy':'epoch','eval_strategy':'epoch',
            'save_total_limit':1,'report_to':'none','plot_loss':False}
        # JSON is valid YAML and removes implicit YAML scalar conversions.
        (args.root/(key+'.yaml')).write_text(json.dumps(config,indent=2))
    print('Prepared all three datasets and explicit learning_rate=1e-4 LLaMA-Factory configs')


if __name__=='__main__': main()
