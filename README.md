# synthetic-data-factory

Turns a handful of seed examples into a large, quality-filtered training dataset.

## Pipeline

1. **Seed** — your 5-20 hand-written examples (input → ideal output).
2. **Expand** — LLM generates paraphrases, edge cases, adversarial variants.
3. **Filter** — each candidate is scored by an independent judge LLM on faithfulness + diversity; low scores dropped.
4. **Dedup** — near-duplicate removal via embedding cosine similarity.
5. **Export** — JSONL ready for fine-tuning (OpenAI / HF format).

## Why this is interesting

Real ML teams spend more time on data than on models. This project is the data half: measurable quality gates, not just "generate 1000 examples and hope".

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m factory.demo
```
