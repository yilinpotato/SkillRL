"""Search-R1 /retrieve adapter; no paid fallback and no benchmark policy changes."""
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit
import urllib.request


class SearchR1Search:
    def __init__(self, audit_path, endpoint, corpus_id, max_results=5):
        parsed = urlsplit(endpoint)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('Explicit Search-R1 HTTP endpoint without credentials required')
        if not corpus_id:
            raise ValueError('Pinned retrieval corpus/index identity required')
        self.endpoint, self.corpus_id = endpoint, corpus_id
        self.audit_path, self.max_results = Path(audit_path), max_results
        self.calls = self.credits = self.missing_usage = 0
        # Cluster traffic must not pass through the external provider proxy.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def search(self, query):
        payload = {'queries': [query], 'topk': self.max_results, 'return_scores': True}
        request = urllib.request.Request(self.endpoint, json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        self.calls += 1
        try:
            with self.opener.open(request, timeout=60) as response:
                data = json.load(response)
        except Exception as exc:
            raise RuntimeError('Search-R1 retrieval failed; paid fallback disabled') from exc
        batches = data.get('result')
        if not isinstance(batches, list) or len(batches) != 1 or not isinstance(batches[0], list):
            raise ValueError('Invalid Search-R1 result batch')
        results = []
        for hit in batches[0][:self.max_results]:
            doc = hit['document']
            contents = doc.get('contents', '')
            title = doc.get('title') or contents.partition('\n')[0].strip('"')
            content = doc.get('text') or contents.partition('\n')[2] or contents
            if not isinstance(content, str) or not content:
                raise ValueError('Search-R1 document has no evidence text')
            results.append({'title': title, 'content': content,
                            'url': 'https://en.wikipedia.org/wiki/' + quote(title.replace(' ', '_'), safe=''),
                            'score': hit.get('score'), 'corpus_id': self.corpus_id})
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'timestamp': datetime.now(timezone.utc).isoformat(),
                'backend': 'search_r1', 'corpus_id': self.corpus_id, 'endpoint': 'retrieve',
                'request': payload, 'response': {'results': results},
                'paid_search_credits': 0}, ensure_ascii=False) + '\n')
        return results


def create_search(config, audit_path, tavily_class):
    backend = config.get('backend', 'tavily')
    if backend == 'tavily':
        return tavily_class(audit_path, max_results=5)
    if backend == 'search_r1':
        return SearchR1Search(audit_path, config.get('retrieval_endpoint', ''),
                              config.get('retrieval_corpus_id', ''), max_results=5)
    raise ValueError('Unknown search backend: ' + str(backend))
