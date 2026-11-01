"""Latch provider failures without retaining credentials or response bodies."""
import json
from pathlib import Path


class CloudAPIGuard:
    def __init__(self, create, output):
        self.create = create
        self.output = Path(output)
        self.failure = None

    def check(self):
        if self.failure is not None:
            raise RuntimeError('Cloud API unavailable; see cloud_api_failure.json')

    def __call__(self, **kwargs):
        self.check()
        try:
            return self.create(**kwargs)
        except Exception as exc:
            self.failure = {'error_type': type(exc).__name__,
                            'http_status': getattr(exc, 'status_code', None),
                            'valid_cloud_update': False}
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / 'cloud_api_failure.json').write_text(
                json.dumps(self.failure), encoding='utf-8')
            self.check()


def install_cloud_guard(cloud, output):
    analyzer = cloud._get_analyzer()
    if analyzer is None:
        raise RuntimeError('Cloud API initialization failed before rollout')
    guard = CloudAPIGuard(analyzer.client.chat.completions.create, output)
    analyzer.client.chat.completions.create = guard
    # Listing models does not detect exhausted balance: require real generation.
    reply = guard(model=analyzer.model, messages=[{'role': 'user', 'content': 'Reply OK.'}],
                  max_tokens=16, temperature=0)
    if not reply.choices or not reply.usage or reply.usage.completion_tokens <= 0:
        raise RuntimeError('Cloud API generation preflight returned no token usage')
    cloud.api_guard = guard
    return guard
