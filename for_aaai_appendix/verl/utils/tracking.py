# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from dataclasses import dataclass
from typing import List, Union


class Tracking:
    supported_backend = ["console", "jsonl"]

    def __init__(
        self,
        project_name,
        experiment_name,
        default_backend: Union[str, List[str]] = "console",
        config=None,
    ):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        unsupported = set(default_backend) - set(self.supported_backend)
        if unsupported:
            raise ValueError(f"Unsupported logger backends: {sorted(unsupported)}")
        self.logger = {}
        if "console" in default_backend:
            from verl.utils.logger.aggregate_logger import LocalLogger

            self.logger["console"] = LocalLogger(print_to_console=True)
        if "jsonl" in default_backend:
            self.logger["jsonl"] = _JsonlLoggingAdapter(
                project_name,
                experiment_name,
            )

    def log(self, data, step, backend=None):
        for name, logger in self.logger.items():
            if backend is None or name in backend:
                logger.log(data=data, step=step)

    def __del__(self):
        for logger in self.logger.values():
            finish = getattr(logger, "finish", None)
            if callable(finish):
                finish()


class _JsonlLoggingAdapter:
    def __init__(self, project_name: str, experiment_name: str):
        path = os.environ.get("JSONL_METRICS_PATH")
        if not path:
            base = os.environ.get("JSONL_METRICS_DIR")
            if not base:
                base = os.path.join("outputs", project_name, experiment_name)
            path = os.path.join(base, "group_metrics.jsonl")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        print(f"[Tracking] jsonl metrics: {self.path}")

    @staticmethod
    def _coerce(value):
        if isinstance(value, (bool, int, float, str)) or value is None:
            return value
        try:
            import numpy as np

            if isinstance(value, (np.floating, np.integer)):
                return value.item()
        except ImportError:
            pass
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return item()
            except (TypeError, ValueError):
                return None
        return None

    def log(self, data, step):
        metrics = {}
        for key, value in data.items():
            coerced = self._coerce(value)
            if coerced is not None or value is None:
                metrics[key] = coerced
        row = {"step": step, "metrics": metrics}
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def finish(self):
        return None


@dataclass
class ValidationGenerationsLogger:
    def log(self, loggers, samples, step):
        return None
