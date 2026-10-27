"""Private-only ModelScope publication and checksum-verified download (SDK 1.36.3).

Credentials come from SDK login or a private token file, never command-line tokens.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(folder, manifest):
    for entry in manifest['files']:
        name = entry['name']
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Manifest must contain only plain filenames')
        path = folder / name
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError('Checksum mismatch or missing file: ' + name)


def require_private(api, repo):
    info = api.get_model(repo)
    if info.get('Visibility') != 1:
        raise RuntimeError('Repository is not confirmed private; refusing upload')
    return info


def check_existing_files(existing, manifest):
    has_weights = any(name.endswith(('.safetensors', '.bin')) for name in existing)
    for entry in manifest['files']:
        old = existing.get(entry['name'])
        if old and (old.get('Size') != entry['bytes'] or old.get('Sha256') != entry['sha256']):
            # ModelScope creates a starter README even without default config.
            # Replace it only in a repository that has no model weights yet.
            if entry['name'] == 'README.md' and not has_weights:
                continue
            raise RuntimeError('Existing version differs; use a new model version: ' + entry['name'])


def anonymous_check(repo):
    import requests
    session = requests.Session()
    session.trust_env = False
    response = session.get(f'https://www.modelscope.cn/api/v1/models/{repo}/repo/files',
        params={'Revision':'master','Recursive':'true'}, timeout=45)
    if response.status_code in (401,403,404):
        return response.status_code
    response.raise_for_status()
    body = response.json()
    if body.get('Success') is False and str(body.get('Code')) in ('401','403','404'):
        return body.get('Code')
    raise RuntimeError('Anonymous access denial not established; inspect access settings')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('operation', choices=['upload','download','verify'])
    p.add_argument('--repo', help='Actual ModelScope namespace/model-name')
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--token-file', type=Path)
    p.add_argument('--revision', default='master')
    p.add_argument('--receipt', type=Path)
    args = p.parse_args()
    if args.operation == 'verify':
        verify(args.folder, json.loads((args.folder/'release-manifest.json').read_text()))
        print('All manifest files verified'); return
    if not args.repo or len(args.repo.split('/')) != 2:
        raise ValueError('Specify the actual namespace/model repository')
    from modelscope.hub.api import HubApi, ModelScopeConfig
    class PrivateApi(HubApi):
        def create_repo(self, repo_id, **kwargs):
            kwargs['visibility'] = 'private'
            kwargs['create_default_config'] = False
            return super().create_repo(repo_id, **kwargs)
    api = PrivateApi(endpoint='https://www.modelscope.cn')
    token = args.token_file.read_text().strip() if args.token_file else os.getenv('MODELSCOPE_API_TOKEN')
    if token:
        api.login(token)
    if not ModelScopeConfig.get_cookies():
        raise RuntimeError('ModelScope login required; do not put tokens in chat or Git')
    if args.operation == 'download':
        from modelscope import snapshot_download
        snapshot_download(args.repo, revision=args.revision, local_dir=str(args.folder))
        verify(args.folder, json.loads((args.folder/'release-manifest.json').read_text()))
        print('Downloaded and verified all manifest files'); return
    manifest = json.loads((args.folder/'release-manifest.json').read_text())
    if args.repo.split('/')[-1] != manifest['model_id']:
        raise ValueError('Repository name differs from model identity')
    verify(args.folder, manifest)
    api.create_repo(args.repo, exist_ok=True)
    require_private(api, args.repo)
    anonymous_check(args.repo)
    existing = {f['Path']:f for f in api.get_model_files(args.repo, recursive=True, use_cookies=True)}
    check_existing_files(existing, manifest)
    names = [item['name'] for item in manifest['files']] + ['release-manifest.json','SHA256SUMS','README.md']
    if not (args.folder/'README.md').is_file():
        raise ValueError('Provide a model card README.md before upload')
    api.upload_folder(repo_id=args.repo, folder_path=str(args.folder), allow_patterns=names,
        max_workers=2, commit_message='Upload verified versioned SFT model (private)')
    require_private(api, args.repo)
    denial = anonymous_check(args.repo)
    files = api.get_model_files(args.repo, recursive=True, use_cookies=True)
    remote = {f['Path']:f for f in files if f.get('Type') != 'tree'}
    for item in manifest['files']:
        found = remote.get(item['name'],{})
        if found.get('Size') != item['bytes'] or found.get('Sha256') != item['sha256']:
            raise RuntimeError('Remote size/hash not verified: ' + item['name'])
    receipt = {'repo':args.repo,'url':'https://www.modelscope.cn/models/'+args.repo,
        'visibility':'private','anonymous_denied':denial,'files_verified':len(manifest['files']),
        'file_revisions':sorted({f['Revision'] for f in remote.values() if f.get('Revision')}),
        'manifest_sha256':sha256(args.folder/'release-manifest.json')}
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
