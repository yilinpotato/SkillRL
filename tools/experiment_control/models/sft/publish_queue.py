"""Finish authorized model uploads after training, with private download verification.

This is a finite release pipeline, not a recurring task. Credentials stay outside Git.
"""
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
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--releases',type=Path,required=True)
    p.add_argument('--token-file',type=Path,required=True)
    p.add_argument('--namespace',required=True)
    p.add_argument('--existing-search-pid',type=int)
    p.add_argument('--github-repo',required=True)
    p.add_argument('--github-branch',required=True)
    p.add_argument('--gh',default='gh')
    args=p.parse_args()
    lock=(args.root/'publish.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    script=Path(__file__).with_name('modelscope_transfer.py')
    status={'updated_at':None,'status':'in_progress','models':{
        'search':{'status':'uploading'},'webshop':{'status':'trained_waiting_upload'},
        'alfworld':{'status':'training'}}}
    saved_status=args.root/'modelscope-status.json'
    if saved_status.exists():
        status=json.loads(saved_status.read_text())
        status['status']='in_progress'
        status.pop('error_type',None)
    training=json.loads((args.root/'queue-status.json').read_text())
    if training.get('jobs',{}).get('alfworld',{}).get('status')=='exported':
        if status['models'].get('alfworld',{}).get('status')!='uploaded_and_verified':
            status['models']['alfworld']['status']='trained_waiting_upload'
    deadline=time.monotonic()+18*3600
    def save():
        status['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        target=args.root/'modelscope-status.json'
        target.write_text(json.dumps(status,indent=2))
        endpoint=f'repos/{args.github_repo}/contents/models/sft/modelscope-status.json'
        try:
            raw=subprocess.check_output([args.gh,'api',endpoint+'?ref='+args.github_branch],stderr=subprocess.DEVNULL)
            previous=json.loads(raw)
            body={'message':'Update private SFT model publication status','branch':args.github_branch,
                  'sha':previous['sha'],'content':base64.b64encode(target.read_bytes()).decode()}
            with tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False) as f:
                json.dump(body,f);name=f.name
            try:
                subprocess.run([args.gh,'api','--method','PUT',endpoint,'--input',name],
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
            finally: os.unlink(name)
        except Exception as exc:
            print('GitHub status update failed: '+type(exc).__name__,flush=True)
    def wait():
        if time.monotonic()>deadline: raise TimeoutError('Release pipeline time limit reached')
        time.sleep(30)
    try:
        for key,display in [('search','Search'),('webshop','WebShop'),('alfworld','ALFWorld')]:
            if status['models'].get(key,{}).get('status')=='uploaded_and_verified':
                continue
            model=f'CoSkill-Qwen3-4B-{display}-SFT-v1';repo=args.namespace+'/'+model
            entry=status['models'][key]={'repo':repo,'url':'https://www.modelscope.cn/models/'+repo,
                'status':'waiting','visibility':'private'}
            save()
            receipt=args.root/(key+'-upload-receipt.json')
            if key=='search' and args.existing_search_pid:
                while not receipt.exists():
                    try: os.kill(args.existing_search_pid,0)
                    except ProcessLookupError: break
                    wait()
            if key=='alfworld':
                while True:
                    training=json.loads((args.root/'queue-status.json').read_text())
                    if training.get('jobs',{}).get('alfworld',{}).get('status')=='exported': break
                    if training.get('status')=='stopped': raise RuntimeError('Training queue stopped; not uploading incomplete model')
                    wait()
            if not receipt.exists():
                entry['status']='uploading';save()
                with (args.root/(key+'-publish.log')).open('w') as log:
                    subprocess.run([sys.executable,str(script),'upload','--repo',repo,
                        '--folder',str(args.releases/model),'--token-file',str(args.token_file),
                        '--receipt',str(receipt)],stdout=log,stderr=subprocess.STDOUT,check=True)
            entry.update(json.loads(receipt.read_text()),status='verifying_download');save()
            from modelscope.hub.file_download import model_file_download
            from modelscope.hub.api import ModelScopeConfig
            from modelscope_transfer import sha256, require_private, anonymous_check
            from modelscope.hub.api import HubApi
            api=HubApi(endpoint='https://www.modelscope.cn')
            require_private(api,repo);anonymous_check(repo)
            manifest=json.loads((args.releases/model/'release-manifest.json').read_text())
            weight=min((f for f in manifest['files'] if f['name'].endswith('.safetensors')),key=lambda f:f['bytes'])
            with tempfile.TemporaryDirectory(prefix='ms-download-check-',dir=args.root) as tmp:
                path=model_file_download(repo,weight['name'],local_dir=tmp,
                    cookies=ModelScopeConfig.get_cookies())
                if sha256(Path(path))!=weight['sha256']: raise ValueError('Downloaded weight checksum mismatch')
            entry.update(status='uploaded_and_verified',download_verified_file=weight['name'],
                download_verified_bytes=weight['bytes'],download_verified_sha256=weight['sha256'])
            save()
        status['status']='completed';save()
    except Exception as exc:
        status.update(status='stopped',error_type=type(exc).__name__)
        save()
        print('Release pipeline stopped: '+type(exc).__name__,flush=True)
        raise SystemExit(1)


if __name__=='__main__': main()
