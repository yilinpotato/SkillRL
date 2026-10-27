"""Finite v2 train/export/private-publish pipeline, never mutating v1 artifacts."""
import argparse
import base64
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    p=argparse.ArgumentParser()
    for key in ('root','releases','token-file'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--upload-python',required=True)
    p.add_argument('--gh',required=True);p.add_argument('--github-repo',required=True)
    p.add_argument('--github-branch',default='main');p.add_argument('--namespace',required=True)
    args=p.parse_args();scripts=Path(__file__).resolve().parent
    sys.path.insert(0,str(scripts.parent));from run_queue import preflight
    lock=(args.root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status={'status':'running','framework':'LLaMA-Factory 0.9.3','learning_rate':1e-4,
            'models':{key:{'status':'queued'} for key in ('search','webshop','alfworld')}}
    def save():
        status['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        dest=args.root/'status.json';tmp=args.root/'status.tmp'
        tmp.write_text(json.dumps(status,indent=2));tmp.replace(dest)
        endpoint=f'repos/{args.github_repo}/contents/models/sft/v2/status.json'
        try:
            prev=json.loads(subprocess.check_output([args.gh,'api',endpoint+'?ref='+args.github_branch],stderr=subprocess.DEVNULL))
            body={'message':'Update LLaMA-Factory SFT v2 progress','branch':args.github_branch,'sha':prev['sha'],
                  'content':base64.b64encode(dest.read_bytes()).decode()}
            with tempfile.NamedTemporaryFile(mode='w',delete=False) as f: json.dump(body,f);path=f.name
            try: subprocess.run([args.gh,'api','--method','PUT',endpoint,'--input',path],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            finally: os.unlink(path)
        except Exception as exc: print('Status sync failed: '+type(exc).__name__,flush=True)
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',HF_HUB_OFFLINE='1')
    def call(command,log):
        with (args.root/log).open('w') as out: subprocess.run(command,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
    try:
        assert (args.root/'preflight.json').exists()
        for key,display in [('search','Search'),('webshop','WebShop'),('alfworld','ALFWorld')]:
            entry=status['models'][key];entry['model_id']=f'CoSkill-Qwen3-4B-{display}-SFT-v2'
            env['CUDA_VISIBLE_DEVICES']=preflight(args.gpu)
            entry['status']='training';save()
            call([sys.executable,'-m','llamafactory.cli','train',str(args.root/(key+'.yaml'))],key+'-train.log')
            entry['status']='exporting';save();env['CUDA_VISIBLE_DEVICES']=''
            call([sys.executable,str(scripts/'export.py'),'--root',str(args.root),'--environment',key,
                  '--output',str(args.releases/entry['model_id'])],key+'-export.log')
            entry['status']='exported';save()
        env['CUDA_VISIBLE_DEVICES']=''
        for key,entry in status['models'].items():
            entry['status']='uploading';save();repo=args.namespace+'/'+entry['model_id']
            receipt=args.root/(key+'-upload-receipt.json')
            call([args.upload_python,str(scripts.parent/'modelscope_transfer.py'),'upload','--repo',repo,
                '--folder',str(args.releases/entry['model_id']),'--token-file',str(args.token_file),
                '--receipt',str(receipt)],key+'-upload.log')
            entry['status']='verifying_download';save()
            call([args.upload_python,str(scripts/'verify_download.py'),'--repo',repo,
                '--folder',str(args.releases/entry['model_id']),'--receipt',str(receipt)],key+'-download.log')
            entry.update(json.loads(receipt.read_text()),status='uploaded_and_verified');save()
        status['status']='completed';save()
    except Exception as exc:
        status.update(status='stopped',error_type=type(exc).__name__);save();raise


if __name__=='__main__': main()
