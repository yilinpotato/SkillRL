"""Use LLaMA-Factory's native trainer without importing unrelated WebUI/vLLM chat engines."""
import json
import sys
from pathlib import Path
from llamafactory.train.tuner import run_exp

if __name__ == '__main__':
    run_exp(args=json.loads(Path(sys.argv[1]).read_text()))
