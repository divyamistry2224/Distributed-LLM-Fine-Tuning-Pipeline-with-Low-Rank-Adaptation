PYTHON ?= python
.PHONY: prepare tiny test smoke distributed sweep serve
prepare:
	$(PYTHON) -m src.prepare --input examples/document_qa.jsonl --output data/processed
tiny: prepare
	$(PYTHON) scripts/build_tiny.py
test:
	$(PYTHON) -m pytest
smoke:
	OMP_NUM_THREADS=1 $(PYTHON) scripts/smoke_test.py
distributed:
	OMP_NUM_THREADS=1 $(PYTHON) scripts/distributed_smoke.py
sweep: prepare
	OMP_NUM_THREADS=1 $(PYTHON) scripts/sweep_ranks.py
serve:
	LORA_BASE_MODEL=examples/artifacts/tiny/base LORA_ADAPTER_DIR=examples/artifacts/tiny/adapter $(PYTHON) -m uvicorn src.api:app --host 0.0.0.0 --port 8000
