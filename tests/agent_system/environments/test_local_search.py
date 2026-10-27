"""HTTP contract tests, not evidence of real Wikipedia retrieval quality."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

source = Path(__file__).resolve().parents[3] / 'agent_system/local_search.py'
spec = importlib.util.spec_from_file_location('local_search_under_test', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LocalSearchTest(unittest.TestCase):
    def test_http_schema_mapping_and_no_auth(self):
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append((self.path, self.headers.get('Authorization'),
                             json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                self.send_response(200); self.end_headers()
                data = {'result': [[{'document': {
                    'contents': '"Eggplant"\nA vegetable used in moussaka.'}, 'score': 2.5}]]}
                if seen[-1][2]['queries'] == ['empty']:
                    data = {'result': [[]]}
                elif seen[-1][2]['queries'] == ['malformed']:
                    data = {'result': []}
                self.wfile.write(json.dumps(data).encode())
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                audit = Path(tmp)/'audit.jsonl'
                client = module.create_search({'backend': 'search_r1',
                    'retrieval_endpoint': f'http://127.0.0.1:{server.server_port}/retrieve',
                    'retrieval_corpus_id': 'fixture-only'}, audit, None)
                results = client.search('moussaka vegetable')
                self.assertEqual(results[0]['content'], 'A vegetable used in moussaka.')
                self.assertEqual(results[0]['title'], 'Eggplant')
                self.assertEqual(seen, [('/retrieve', None, {'queries': ['moussaka vegetable'], 'topk': 5, 'return_scores': True})])
                self.assertEqual(client.credits, 0)
                self.assertEqual(json.loads(audit.read_text())['backend'], 'search_r1')
                # Exercise the actual benchmark action loop against the HTTP adapter.
                protocol_path = source.parent / 'environments/benchmark_protocol.py'
                protocol_spec = importlib.util.spec_from_file_location('protocol_under_test', protocol_path)
                protocol = importlib.util.module_from_spec(protocol_spec)
                protocol_spec.loader.exec_module(protocol)
                env = protocol.SearchQAEnvironment('What vegetable?', client, 15, 8)
                observation, valid = env.step('<think>Find evidence.</think><action>search[moussaka vegetable]</action>')
                self.assertTrue(valid)
                self.assertIn('A vegetable used in moussaka.', observation)
                self.assertEqual(env.searches, 1)
                _, valid = env.step('<think>Read page.</think><action>open[1]</action>')
                self.assertFalse(valid)
                _, valid = env.step('<think>Evidence found.</think><action>answer[eggplant]</action>')
                self.assertTrue(valid and env.done)
                self.assertEqual(protocol.score_answer(env.answer, ['eggplant'])['exact_match'], 1.)
                self.assertEqual(client.search('empty'), [])
                with self.assertRaises(ValueError):
                    client.search('malformed')
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_missing_provenance_rejected(self):
        with self.assertRaises(ValueError):
            module.create_search({'backend': 'search_r1', 'retrieval_endpoint': 'http://localhost/retrieve'}, 'unused', None)

    def test_no_implicit_paid_fallback(self):
        with self.assertRaises(ValueError):
            module.create_search({'backend': 'unknown'}, 'unused', lambda *a, **k: self.fail('paid fallback'))


if __name__ == '__main__':
    unittest.main()
