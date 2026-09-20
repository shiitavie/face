# Local setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,local,docs]"
.venv/bin/python -m pytest tests/ -q
```

Then run everything through `.venv/bin/python` — the editable install puts
`facecav` on the path, so no `PYTHONPATH` is needed.

## Extras

| extra   | contents        | when |
|---------|-----------------|------|
| `dev`   | pytest          | always, locally |
| `local` | torch           | locally — **never on Colab**, it would replace the CUDA-matched build |
| `docs`  | python-docx     | only to regenerate the Word manuscript summary |

Core dependencies deliberately exclude `torch`. Colab installs with
`pip install -e ".[dev]"` and keeps its own CUDA-matched build.

## Anthropic credentials

The Claude battery needs inference access. Either works:

```bash
ant auth login          # OAuth profile; anthropic>=1.0 reads it automatically
export ANTHROPIC_API_KEY=sk-ant-...
```

Verify:

```bash
.venv/bin/python -c "import anthropic; print(anthropic.Anthropic().messages.count_tokens(
    model='claude-opus-5', messages=[{'role':'user','content':'hi'}]).input_tokens)"
```

`count_tokens` is free, so this proves auth without spending anything.

## Data

CFD is licensed and not in this repository. Obtain it from chicagofaces.org and
point scripts at it with `--cfd-root`, or place it at
`dataset/CFD Version 3.0/` (gitignored).
