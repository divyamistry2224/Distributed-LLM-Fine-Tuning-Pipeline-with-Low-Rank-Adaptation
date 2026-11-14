# LoRA Lab

A buildless public interface for the fine-tuning project. Five workbench sections provide distinct interactions:

- Adapter playground: adjustable rank, illustrative factor matrices, parameter savings and a signal animation.
- Document studio: real, local pretrained DistilBERT QA using Transformers.js and ONNX Runtime / WebAssembly. First use downloads public q8 weights (approximately 65 MB). Questions and context stay in the browser. The input budget is 512 model tokens.
- Experiment explorer: recorded rank experiments, parameter counts, held-out metrics and downloadable JSON.
- Recovery lab: an explicitly labeled checkpoint-protocol simulation, resume-contract checks and actual document split inspection.
- Architecture: inspect clickable pipeline nodes, trace an illustrative request path and download the full SVG.

The browser QA model is a pretrained extractive baseline, not the project's LoRA adapter. LoRA measurements come from the tiny CPU fixture and do not establish QA quality. This distinction is visible in the interface.

## Hosting

Serve this directory as static files. GitHub Pages deployment is prepared in `.github/workflows/pages.yml`; a repository administrator must first select **Settings → Pages → Source → GitHub Actions**, then run **Publish experiment explorer** from Actions. The connected GitHub integration cannot enable Pages for this repository.

No API keys or server are needed for the static demo. CDN access to jsDelivr, Hugging Face model files and Google Fonts is used; fonts have local fallbacks.

## Checks

```bash
npm ci --ignore-scripts
npm test
```

DOM tests verify navigation, rank/metric filters, actual split counts, recovery transitions, contract rejection and QA status/error handling. They mock the worker transport and do not verify rendering or browser inference. A separate real model execution is used during deployment verification.
