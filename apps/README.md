# Apps

Project-facing interfaces live here when they grow beyond one-off scripts.

Current boundaries:

- `agent/`: Pi Agent adapters for the validated kernel-development loop.
- `modal/`: Modal deployment code if `scripts/modal_vllm.py` becomes more than a single deployment file.

Keep small operational scripts in `scripts/` until they need package-local assets,
tests, or multiple modules.
