# Backend Launchers

These scripts are implementation details of the public `../run.sh` interface:

- `run_norl_alfworld.sh`: frozen ALFWorld executor;
- `run_norl_webshop.sh`: frozen WebShop executor;
- `run_tree_rl.sh`: common Ray/GRPO launcher for ALFWorld and WebShop.

Direct invocation is supported for debugging, but reported experiments should
use `run.sh` so paths, benchmark selection, and RL mode are recorded
consistently.
