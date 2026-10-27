"""Audit actual LLaMA-Factory preprocessing before training a selected SFT job."""
import argparse
import json
from pathlib import Path


def main():
    import ijson
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    p=argparse.ArgumentParser(); p.add_argument('config',type=Path); a=p.parse_args()
    cfg=json.loads(a.config.read_text()); cfg.update(use_cpu=True,bf16=False)
    model_args,data_args,training_args,_,_=get_train_args(cfg)
    ts=load_tokenizer(model_args); tok=ts['tokenizer']
    template=get_template_and_fix_tokenizer(tok,data_args)
    ds=get_dataset(template,model_args,data_args,training_args,stage='sft',**ts)
    report={}
    for split,name in [('train','train_dataset'),('validation','eval_dataset')]:
        source_path=Path(cfg['dataset_dir'])/f'{split}.json'
        source_handle=source_path.open('rb')
        original=ijson.items(source_handle,'item')
        encoded=ds[name]
        lengths=[]
        for source,row in zip(original,encoded):
            messages=source['messages']; system=''
            if messages[0]['role']=='system': system=messages[0]['content']; messages=messages[1:]
            pairs=template.encode_multiturn(tok,messages,system)
            full=sum(len(x)+len(y) for x,y in pairs)
            if full != len(row['input_ids']) or full>cfg['cutoff_len']:
                raise ValueError(f'Truncation detected: {split}, {full} tokens')
            assert [v for v in row['labels'] if v!=-100]==pairs[-1][1], 'Wrong supervision mask'
            lengths.append(full)
        assert len(lengths)==len(encoded) and next(original,None) is None, 'Dropped samples'
        source_handle.close()
        report[split]={'samples':len(encoded),'max_tokens':max(lengths),'no_truncation':True,'final_response_only':True}
    a.config.with_suffix('.audit.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report))


if __name__=='__main__': main()
