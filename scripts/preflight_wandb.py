#!/usr/bin/env python3
"""Verify that an online Weights & Biases run can be created before training.

The command intentionally never prints credentials.  It uses the same
environment-based login path as ``wandb.init`` in the verl tracker.
"""

from __future__ import annotations

import os
import socket
import sys


def _is_network_failure(exc: BaseException) -> bool:
    """Conservatively recognize failures where W&B cannot be reached at all."""
    if isinstance(exc, (ConnectionError, TimeoutError, socket.gaierror)):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "connection refused",
            "connection reset",
            "network is unreachable",
            "name or service not known",
            "temporary failure in name resolution",
            "timed out",
        )
    )


def main() -> int:
    mode = os.environ.get("WANDB_MODE", "online").lower()
    if mode != "online":
        print(
            f"[wandb-preflight] FAIL WANDB_MODE must be 'online' for realtime tracking, got {mode!r}",
            file=sys.stderr,
        )
        return 2
    try:
        import wandb

        # ``verify=True`` performs a real authenticated request.  ``anonymous``
        # is deliberately disabled so a missing login cannot silently create a
        # different anonymous run.
        logged_in = wandb.login(anonymous="never", relogin=False, verify=True)
    except Exception as exc:
        if _is_network_failure(exc):
            print(
                "[wandb-preflight] OFFLINE_FALLBACK online W&B is unreachable; "
                "switching to offline mode is safe.",
                file=sys.stderr,
            )
            return 10
        print(
            f"[wandb-preflight] FAIL online authentication: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 3
    if not logged_in:
        print("[wandb-preflight] FAIL W&B did not report an authenticated login.", file=sys.stderr)
        return 4
    print(
        "[wandb-preflight] OK "
        f"project={os.environ.get('WANDB_PROJECT', 'coskill-tree-rl')} "
        f"entity={os.environ.get('WANDB_ENTITY', '<default>')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
