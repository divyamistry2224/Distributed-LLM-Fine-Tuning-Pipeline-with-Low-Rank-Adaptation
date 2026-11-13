# Architecture and execution contracts

## Data boundary

`src.prepare` validates nonempty context/question/answer fields and records rejected lines in quarantine. Duplicate QA pairs and conflicting answers are quarantined. A union-find groups shared document IDs and normalized contexts before seeded splitting; aliases therefore cannot bridge development and test data. The manifest records input/output hashes and split counts.

## Supervision boundary

Tokenization reserves space for the complete question, answer and EOS. Only document context is truncated. Impossible token budgets raise errors before training. Labels mask prompt and padding positions by index, preserving EOS supervision even when PAD and EOS share an ID.

## Distributed training

Accelerate prepares loaders, models and optimizers. Scheduler steps follow actual optimizer updates, including partial accumulation windows. DDP synchronizes adapters; a cross-rank parameter hash checks agreement before export. Validation gathers per-example NLL sums and supervised-token counts, removing distributed tail duplication before aggregation.

Checkpoints contain Accelerate optimizer/scheduler/RNG state, adapter weights, tokenizer, epoch/batch cursor and a compatibility contract. Resume requires the same data hashes, model reference, world size, hyperparameters and planned update count. Exact resumed equality is verified on the zero-dropout CPU fixture; this is not a guarantee of bitwise reproducibility on all hardware.

QLoRA assigns each CUDA rank its local quantized model. Accelerate owns placement for nonquantized training. No automatic model sharding is mixed with DDP. FSDP and DeepSpeed are not supported by this loop.

## Evaluation and delivery

Rank sweeps select using validation NLL without consuming test records. Test evaluation checks development overlap and reports frozen-base and adapter metrics with individual predictions. NLL is computed on answer tokens only; generation metrics use normalized exact match and bag-of-token F1.

FastAPI loads one engine at startup and exposes readiness plus constrained generation. The engine serializes calls to a single model. Optional bearer authentication protects generation; deployment operators must provide transport security, rate controls and resource limits appropriate to their host. Adapter merging creates a standalone model using a CPU safe-merge path.

## Automation

GitHub Actions runs lint, unit tests, a real single-process train/resume/evaluate/API/merge check and a two-worker CPU/Gloo run. JSON execution reports are uploaded as workflow artifacts. Tiny local fixtures avoid network dependencies in integration tests. The Dockerfile packages CPU inference; CUDA training uses a separately installed PyTorch runtime.
