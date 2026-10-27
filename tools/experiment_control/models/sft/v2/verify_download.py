"""Authenticated fresh download check after a private upload."""
import argparse
import json
import sys
import tempfile
from pathlib import Path


def main():
    from modelscope.hub.api import HubApi,ModelScopeConfig
    from modelscope.hub.file_download import model_file_download
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
    from modelscope_transfer import require_private,anonymous_check,sha256
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True)
    p.add_argument('--folder',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args();api=HubApi(endpoint='https://www.modelscope.cn')
    require_private(api,args.repo);anonymous_check(args.repo)
    manifest=json.loads((args.folder/'release-manifest.json').read_text())
    item=min((x for x in manifest['files'] if x['name'].endswith('.safetensors')),key=lambda x:x['bytes'])
    with tempfile.TemporaryDirectory(prefix='sft-v2-download-',dir=args.receipt.parent) as tmp:
        path=model_file_download(args.repo,item['name'],local_dir=tmp,cache_dir=str(Path(tmp)/'cache'),
                                 cookies=ModelScopeConfig.get_cookies())
        assert sha256(Path(path))==item['sha256']
    result=json.loads(args.receipt.read_text());result.update(download_verified_file=item['name'],
        download_verified_bytes=item['bytes'],download_verified_sha256=item['sha256'])
    args.receipt.write_text(json.dumps(result,indent=2))


if __name__=='__main__': main()
