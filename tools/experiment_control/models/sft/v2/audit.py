"""Exercise LLaMA-Factory's real tokenizer/processor on every retained sample."""
import argparse
import json
from pathlib import Path


def main():
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    report={}
    for key in ('search','webshop','alfworld'):
        config=json.loads((args.root/(key+'.yaml')).read_text())
        config.update(use_cpu=True,bf16=False)
        model_args,data_args,training_args,finetuning_args,generating_args=get_train_args(config)
        tokenizers=load_tokenizer(model_args);tokenizer=tokenizers['tokenizer']
        template=get_template_and_fix_tokenizer(tokenizer,data_args)
        datasets=get_dataset(template,model_args,data_args,training_args,stage='sft',**tokenizers)
        results={}
        for split,name in [('train','train_dataset'),('validation','eval_dataset')]:
            original=json.loads((args.root/(key+'-data')/(split+'.json')).read_text())
            encoded=datasets[name]
            assert len(original)==len(encoded),(key,split,'dropped samples')
            max_length=0;supervised=0
            for source,row in zip(original,encoded):
                messages=source['messages'];system=''
                if messages[0]['role']=='system': system=messages[0]['content'];messages=messages[1:]
                pairs=template.encode_multiturn(tokenizer,messages,system)
                full=sum(len(a)+len(b) for a,b in pairs)
                assert full==len(row['input_ids']) and full<=config['cutoff_len'],(key,'truncation')
                target=[x for x in row['labels'] if x!=-100]
                assert target==pairs[-1][1],(key,'supervision must be final assistant only')
                decoded=tokenizer.decode(target,skip_special_tokens=False)
                assert messages[-1]['content'] in decoded,(key,'target text changed')
                max_length=max(max_length,full);supervised+=len(target)
            results[split]={'samples':len(encoded),'max_tokens':max_length,'supervised_tokens':supervised,
                'no_truncation':True,'final_response_only':True,'response_preserved':True}
        report[key]=results
    (args.root/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__': main()
