"""Launch the pinned upstream BM25 implementation without modifying its ranking."""
import argparse
import importlib.util
from pathlib import Path
import subprocess

UPSTREAM_COMMIT = '598e61bd1d36895726d28a8d06b3a15bed19f5d3'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8001)
    args = parser.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError('Search-R1 checkout does not match pinned commit')
    relative = 'search_r1/search/retrieval_server.py'
    if subprocess.check_output(['git', '-C', str(args.repo), 'status', '--porcelain', '--', relative], text=True).strip():
        raise ValueError('Upstream retrieval source has local changes')
    if not args.index.is_dir() or not args.corpus.is_file():
        raise ValueError('Verified index directory and unpacked corpus required')
    spec = importlib.util.spec_from_file_location('search_r1_upstream', args.repo / relative)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    upstream.config = upstream.Config(retrieval_method='bm25', retrieval_topk=5,
        index_path=str(args.index), corpus_path=str(args.corpus), faiss_gpu=False)
    upstream.retriever = upstream.get_retriever(upstream.config)
    upstream.uvicorn.run(upstream.app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
