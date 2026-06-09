# Apps

Project-facing interfaces live here when they grow beyond one-off scripts.

Planned boundaries:

- `agent/`: adapters for the validated kernel-development loop. Pi Agent is the
  current candidate to study before adding code here.
- `modal/`: Modal deployment code if `scripts/modal_vllm.py` becomes more than a single deployment file.

Keep small operational scripts in `scripts/` until they need package-local assets,
tests, or multiple modules.
