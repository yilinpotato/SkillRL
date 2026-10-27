"""Finite single-job SFT queue with mandatory audit, smoke, durable status and no GPU remapping."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--worker',required=True); p.add_argument('--jobs',nargs='+',required=True)
    a=p.parse_args(); root=a.root
    lock=(root/f'{a.worker}.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'worker':a.worker,'stage':'SFT','jobs':{},'status':'running'}
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',
             TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
    def save():
        state['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        temp=root/f'{a.worker}.status.tmp'; temp.write_text(json.dumps(state,indent=2))
        temp.replace(root/f'{a.worker}.status.json')
    def call(cmd,log):
        with (root/'logs'/log).open('a') as f:
            subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    save()
    try:
        for key in a.jobs:
            state['jobs'][key]={'status':'auditing'}; save()
            config=root/f'{key}.yaml'; cfg=json.loads(config.read_text())
            if not config.with_suffix('.audit.json').exists():
                call([sys.executable,str(Path(__file__).with_name('audit.py')),str(config)],key+'-audit.log')
            output=Path(cfg['output_dir'])
            if output.exists(): raise ValueError('Existing training output requires explicit resume review')
            smoke=dict(cfg,max_steps=2,do_eval=False,eval_strategy='no',save_strategy='steps',save_steps=2,
                       output_dir=str(root/'results'/f'{key}-smoke'),report_to='none')
            smoke.pop('eval_dataset',None)
            smoke_path=root/f'{key}-smoke.yaml'; smoke_path.write_text(json.dumps(smoke,indent=2))
            state['jobs'][key]['status']='smoke';save()
            call([sys.executable,str(Path(__file__).with_name('train_entry.py')),str(smoke_path)],key+'-smoke.log')
            checkpoints=list(Path(smoke['output_dir']).glob('**/adapter_model.safetensors'))
            if not checkpoints: raise RuntimeError('Smoke did not save an adapter')
            state['jobs'][key]['status']='training';save()
            call([sys.executable,str(Path(__file__).with_name('train_entry.py')),str(config)],key+'-train.log')
            final=json.loads((output/'trainer_state.json').read_text())
            if final['epoch']<2.999: raise RuntimeError('Incomplete SFT epochs')
            state['jobs'][key].update(status='trained_adapter',optimizer_steps=final['global_step'],epochs=final['epoch'])
            save()
        state['status']='completed';save()
    except BaseException as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc));save();raise


if __name__=='__main__': main()
