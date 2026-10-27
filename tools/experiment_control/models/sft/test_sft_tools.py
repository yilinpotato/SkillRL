import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from modelscope_transfer import require_private, verify, check_existing_files


class Tests(unittest.TestCase):
    def test_starter_readme_allowed_only_before_weights_exist(self):
        manifest={'files':[{'name':'README.md','bytes':5,'sha256':'new'}]}
        existing={'README.md':{'Size':3,'Sha256':'old'}}
        check_existing_files(existing,manifest)
        existing['model.safetensors']={'Size':1,'Sha256':'weight'}
        with self.assertRaises(RuntimeError): check_existing_files(existing,manifest)
        with self.assertRaises(RuntimeError):
            check_existing_files({'config.json':{'Size':1,'Sha256':'old'}},
                {'files':[{'name':'config.json','bytes':2,'sha256':'new'}]})

    def test_private_gate_rejects_public_and_unknown(self):
        class Api:
            def get_model(self, repo):
                return self.info
        api = Api()
        for info in ({'Visibility':5}, {}, {'Visibility':None}, {'Visibility':'private'}):
            api.info = info
            with self.assertRaises(RuntimeError):
                require_private(api, 'owner/model')
        api.info = {'Visibility':1}
        self.assertEqual(require_private(api,'owner/model'),api.info)

    def test_manifest_detects_same_size_corruption_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); (folder/'config.json').write_bytes(b'good')
            manifest={'files':[{'name':'config.json','bytes':4,'sha256':hashlib.sha256(b'good').hexdigest()}]}
            verify(folder,manifest)
            (folder/'config.json').write_bytes(b'evil')
            with self.assertRaises(ValueError): verify(folder,manifest)
            manifest['files'][0]['name']='../outside'
            with self.assertRaises(ValueError): verify(folder,manifest)

    def test_jsonl_unicode_separator_is_not_a_record_boundary(self):
        text=json.dumps({'text':'a\u2028b'},ensure_ascii=False)+'\n'
        rows=[json.loads(line) for line in text.split('\n') if line.strip()]
        self.assertEqual(rows,[{'text':'a\u2028b'}])

    def test_gpu_guard_rejects_compute_process_even_with_low_reported_memory(self):
        import run_queue
        with patch('run_queue.subprocess.check_output',side_effect=['GPU-test, 36, 0\n','GPU-test, 1234\n']):
            with self.assertRaises(RuntimeError): run_queue.preflight(0)
        with patch('run_queue.subprocess.check_output',side_effect=['GPU-test, 36, 0\n','GPU-other, 999\n']):
            self.assertEqual(run_queue.preflight(0),'GPU-test')


if __name__=='__main__': unittest.main()
