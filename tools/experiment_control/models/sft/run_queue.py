"""Sequential single-GPU training. Refuses occupied devices; never kills other jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def preflight(gpu):
    output = subprocess.check_output(['nvidia-smi','-i',str(gpu),
        '--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    uuid, memory, utilization = [x.strip() for x in output.strip().split(',')]
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader'],text=True)
    if int(memory)>256 or int(utilization)>5 or any(uuid in line for line in processes.splitlines()):
        raise RuntimeError(f'GPU {gpu} is occupied; no processes were changed')
    return uuid


def main():
    import fcntl  # Linux-only process lock; CPU helpers remain importable on Windows.
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--base',required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--releases',type=Path,required=True)
    args=p.parse_args(); scripts=Path(__file__).resolve().parent
    lock=(args.root/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'pid':os.getpid(),'gpu':args.gpu,'started_at':time.time(),'jobs':{}}
    def save():
        temp=args.root/'queue-status.tmp';temp.write_text(json.dumps(state,indent=2))
        temp.replace(args.root/'queue-status.json')
    try:
        for environment, display in [('webshop','WebShop'),('alfworld','ALFWorld')]:
            run=args.root/(environment+'-train'); name=f'CoSkill-Qwen3-4B-{display}-SFT-v1'
            state['jobs'][environment]={'status':'checking_gpu'};save()
            uuid=preflight(args.gpu)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1')
            state['jobs'][environment].update(status='training',started_at=time.time());save()
            with (args.root/(environment+'-train.log')).open('w') as log:
                subprocess.run([sys.executable,str(scripts/'train_skillrl_sft.py'),
                    '--model',args.base,'--data-dir',str(args.root/(environment+'-data')),
                    '--output-dir',str(run)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            state['jobs'][environment]['status']='exporting_cpu';save()
            env['CUDA_VISIBLE_DEVICES']=''
            with (args.root/(environment+'-export.log')).open('w') as log:
                subprocess.run([sys.executable,str(scripts/'export_model.py'),'--run-dir',str(run),
                    '--output',str(args.releases/name),'--name',name],env=env,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            state['jobs'][environment].update(status='exported',finished_at=time.time(),model=name);save()
        state['status']='completed';save()
    except Exception as exc:
        state.update(status='stopped',error=type(exc).__name__+': '+str(exc));save()
        raise


if __name__=='__main__':
    main()
