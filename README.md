# What's In My Human Feedback (WIMHF)

WIMHF learns human-interpretable concepts from preference datasets in four steps:
- encode response texts using an embedding model,
- train a sparse autoencoder (SAE) on the difference in response embeddings across all preference pairs,
- interpret each SAE feature using an LLM (OpenAI or vLLM),
- identify features that predict preference labels.

**Links:** [Paper](https://arxiv.org/abs/2510.26202) · [Demo](https://rajivmovva.com/demo-wimhf/) · [Code](https://github.com/rmovva/wimhf) · [Data](https://huggingface.co/datasets/rmovva/wimhf-data)

Read the preprint for full details: [What’s In My Human Feedback? Learning Interpretable Descriptions of Preference Data](https://arxiv.org/abs/2510.26202) by Rajiv Movva, Smitha Milli, Sewon Min, and Emma Pierson.

## Quickstart

1. **Clone & install**
   ```bash
   git clone https://github.com/rmovva/wimhf.git
   cd wimhf
   pip install -e .
   ```
2. **Configure credentials**  
   Export your OpenAI-compatible key once per shell:
   ```bash
   export OAI_WIMHF=sk-your-key
   ```
   Local LLM inference is supported through `vllm`/`sentence-transformers`; skip the key if you only use those paths.
3. **Prepare a dataset config** *(CLI path)*  
   Copy one of the JSON files in `configs/` and point it at your dataset (see schema below). You can then run:
   ```bash
   python scripts/run_wimhf.py configs/community_align.json --output-dir outputs/community_align
   ```
4. **Open the notebook** *(interactive path)*  
   Alternatively, use `notebooks/community_alignment_quickstart.ipynb` for an end-to-end walkthrough that mirrors the same dataclasses while letting you inspect intermediate artefacts.

## Dataset schema

Provide a table (Parquet/JSON/CSV) with at least the following columns:
- `prompt`: text shown to both models/annotators,
- `response_A`, `response_B`: the two candidate completions,
- `label`: binary or {0, 1} preference target (1 means `response_A` preferred).  

Optional columns include `conversation_id`, `split_columns` for connected-component train/val splits, and derived statistics like `length_delta`. The quickstart utilities will compute `length_delta` automatically if it is missing.

See `configs/*.json` for concrete settings used in the WIMHF paper; each config mirrors the dataclasses in `wimhf.quickstart`.

## API keys and LLM usage

Remote interpretation, annotation, and embedding calls expect the environment variable `OAI_WIMHF`. The library now reads **only** this key to initialise the OpenAI client. Local inference routes are available through:
- `wimhf.llm_local` (via `vllm`) for decoder-only models,
- `wimhf.embedding.get_local_embeddings` (via `sentence-transformers`) for offline embeddings.

Set `CUDA_VISIBLE_DEVICES` when running local models if multiple GPUs are present.

## Citation

If you use this code, please cite:

> What’s In My Human Feedback? Learning Interpretable Descriptions of Preference Data. Rajiv Movva, Smitha Milli, Sewon Min, and Emma Pierson. arXiv:2510.26202.

```
@misc{movva_wimhf_2025,
  title         = {What's In My Human Feedback? Learning Interpretable Descriptions of Preference Data},
  author        = {Rajiv Movva and Smitha Milli and Sewon Min and Emma Pierson},
  year          = {2025},
  eprint        = {2510.26202},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2510.26202}
}
```
