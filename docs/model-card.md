# Dataset and model notes

## Included dataset

`examples/document_qa.jsonl` contains 24 synthetic, independent document-QA examples about software and machine-learning concepts. It is a small engineering fixture distributed under the repository MIT license. It is not representative of real document understanding, sensitive business documents or benchmark performance. Document-group splitting produces 16 training, four validation and four test examples with the default seed.

## Included model

`examples/artifacts/tiny/base` is a randomly initialized two-layer GPT-2 architecture with a locally trained byte-level BPE tokenizer. Vocabulary construction uses training records only. The rank-four adapter targets fused attention projections. The tokenizer, base and adapter are small enough to run offline on CPU.

Measured answer exact match and token F1 on the four held-out examples are zero. Use this artifact to inspect training, checkpoint recovery, loading, merging and HTTP integration; it is not suitable for answering real questions. Slight perplexity changes on this fixture do not establish task improvement.

## Training with pretrained models

The default launch configuration targets a downloadable Qwen base with attention-projection LoRA modules. Confirm the upstream model license, permitted use, tokenizer compatibility and hardware requirements before substituting a pretrained model. Training manifests identify the model reference and dataset fingerprints. For immutable provenance, use a local snapshot pinned to an upstream revision.

Evaluate domain-specific quality on independent documents before deployment. Generated answers can be incorrect or unsupported by the supplied context. This implementation provides document conditioning, not retrieval or factuality guarantees.
