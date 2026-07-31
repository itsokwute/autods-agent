# AutoDS-Agent

**[▶ Try the live demo](https://huggingface.co/spaces/YOUR-USERNAME/autods-agent)** — replace this link with your Space URL after deploying (see `STAGE-4.md`).

**An autonomous data science engineer.** Drop in a CSV, a ZIP of spreadsheets, a PDF full of
tables, or a URL — AutoDS-Agent profiles it, defines the ML problem, writes and *executes*
the analysis in a live Python session, repairs its own errors, trains models, and hands you
a runnable Google Colab notebook plus a written conclusion.

```
Ingest anything  →  Scope the problem  →  Execute & self-correct  →  Model  →  Report  →  Export .ipynb
```

---

## Why this exists

Most "AI data scientist" demos generate code that has never been run. AutoDS-Agent closes the
loop: every cell is executed in a stateful REPL, and failures are fed back to the model with
the full traceback and a live variable inventory until the cell runs clean. What ships in the
exported notebook is code that **actually executed against your data**.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│  app.py — Streamlit dashboard                                         │
│  uploads · URLs · preview · metrics · live stream · plots · export    │
└───────────────┬────────────────────────────┬──────────────────────────┘
                │                            │
   ┌────────────▼──────────────┐   ┌─────────▼─────────────────────────┐
   │ ingestion.py              │   │ agent.py                          │
   │ CSV/XLSX/JSON/YAML/       │   │ ClaudeClient  ── Messages API      │
   │ PDF/DOCX/ZIP/Parquet/URL  │──▶│ PythonREPL    ── stateful exec     │
   │ → dict[str, DataFrame]    │   │ 6-stage pipeline + self-repair     │
   │ + text artifacts          │   │ → spec · RESULTS · report          │
   └───────────────────────────┘   └─────────┬─────────────────────────┘
                                             │
                                   ┌─────────▼─────────────────────────┐
                                   │ notebook_builder.py               │
                                   │ transcript + embedded data        │
                                   │ → executable Colab .ipynb         │
                                   └───────────────────────────────────┘
```

### The pipeline

| Stage | What happens | LLM? |
|---|---|---|
| **1 · Profile** | Deterministic schema, dtype, cardinality and missingness profiling | no |
| **2 · Analyze** | Strict-JSON spec: problem statement, ML paradigm, target, feature typing, leakage risks, metric, validation strategy | yes |
| **3 · EDA** | Generated exploratory code, executed; 3–5 figures captured | yes + exec |
| **4 · Prepare** | Cleaning, imputation, encoding, feature engineering, `ColumnTransformer` inside a `Pipeline`, train/test split | yes + exec |
| **5 · Model** | Dummy + linear/forest baselines, LightGBM, cross-validation, metrics into `RESULTS`, importance and diagnostic plots | yes + exec |
| **6 · Report** | "Conclusion & Business Impact" + "Future Work", grounded strictly in the execution logs | yes |

### Self-correction loop

When a generated cell raises, the agent sends back the trimmed traceback, the partial stdout,
and a live inventory of every variable in the session, then asks for the *complete* corrected
cell. Default budget is 3 repairs per stage (configurable, 0–6). Every repair is logged and
surfaced in the report appendix, so the run stays auditable.

---

## Ingestion coverage

| Input | Handling |
|---|---|
| `.csv` `.tsv` `.txt` | Delimiter sniffing, bad-line skipping |
| `.xlsx` `.xls` `.xlsm` | Every sheet becomes its own table |
| `.json` `.jsonl` | Recursive walk — records, dict-of-lists, and nested containers are all harvested |
| `.yaml` `.yml` | Multi-document safe-load, same recursive harvesting |
| `.pdf` | `pdfplumber` table extraction per page + full text as planner context |
| `.docx` | Tables → DataFrames, paragraphs → context |
| `.zip` | Recursive unpacking (depth 3, 250 members), each member parsed by type |
| `.parquet` | Direct via pyarrow |
| URL | Downloaded, typed by extension or `Content-Type` |

Every table passes through `clean_dataframe`: de-duplicated and de-`Unnamed:`-ed columns,
whitespace stripping, empty-row/column drops, and conservative numeric/datetime inference
(`"1,234"`, `"45%"`, `"$9.99"` all become numbers). A single bad file never sinks a batch —
failures land in a warnings list.

---

## Quickstart

> **Starting from nothing? Read [`BUILD.md`](BUILD.md) first** — it covers what you're
> actually building, assembling the downloaded files into a project folder, and connecting
> the API key.
>
> **Have the folder already? Read [`SETUP.md`](SETUP.md)** — it walks through install, first run,
> GitHub, and deployment step by step, with fixes for the errors you'll actually hit.

```bash
git clone https://github.com/<you>/AutoDS-Agent.git
cd AutoDS-Agent

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY="sk-ant-..."                # or paste it in the sidebar
streamlit run app.py
```

Then: **Load data → Ingest → Agent run → Export**.

### Configuration

| Setting | Default | Notes |
|---|---|---|
| Planner model | `claude-sonnet-5` | The original brief specified Claude 3.7 Sonnet; that snapshot is legacy, so a current model is the default and `claude-3-7-sonnet-latest` is kept in the dropdown for parity. |
| Self-correction attempts | 3 | Per stage |
| Per-cell timeout | 600 s | Soft timeout — see the caveat below |
| Random state | 42 | Threaded into every seeded call, including the exported notebook |
| LightGBM | on | Turn off for a scikit-learn-only run |
| Business goal | — | Free text; steers target selection and the report's framing |
| Force target column | — | Overrides the agent's target choice |

---

## The exported notebook

One click produces an nbformat v4.5 notebook containing:

1. Run metadata — model, run ID, seed, stated goal
2. A `%pip install` bootstrap cell for Colab
3. **Every ingested table embedded as gzip+base64 CSV** (12 MB/table, 30 MB total budget;
   oversized tables get a `google.colab.files.upload()` fallback cell)
4. The problem specification as a formatted Markdown section
5. Every executed cell, in order, with its stage heading and a note where self-correction fired
6. The final `RESULTS` metrics
7. The Conclusion & Business Impact / Future Work report
8. A self-correction audit appendix

Upload to Colab and hit **Runtime → Run all**. No external files, no app required.

---

## ⚠️ Security

**The REPL executes model-generated code inside the Streamlit process.**

`PythonREPL.screen()` statically blocks `subprocess`, `os.system`, `os.popen`, `os.remove`,
`shutil.rmtree`, `eval`, `exec`, `__import__`, runtime `pip install`, and (unless explicitly
enabled) network calls. **This is a speed bump, not a sandbox** — a determined prompt injection
in an uploaded document could work around it.

For anything beyond a local, single-user, trusted-data setup:

- Run inside a container with no credentials mounted and no outbound network beyond
  `api.anthropic.com`
- Mount data read-only; give the container a hard memory and CPU cap
- Never point it at production databases or a machine holding secrets
- Set `APP_PASSWORD` on any internet-reachable deployment (the app gates on it automatically)
- Treat uploaded PDFs and DOCX files as untrusted input — their text reaches the planner prompt

**Known limitation:** the per-cell timeout is enforced by joining a worker thread. Python
cannot safely kill a running thread, so a runaway cell is *reported* as timed out but keeps
consuming CPU until it finishes. Container-level resource limits are the real backstop.

---

## Project layout

```
AutoDS-Agent/
├── app.py                 # Streamlit UI: ingest, stream, plots, report, export
├── ingestion.py           # Universal parsers → dict[str, DataFrame] + text artifacts
├── agent.py               # ClaudeClient, PythonREPL, prompts, 6-stage pipeline
├── notebook_builder.py    # nbformat v4.5 export with embedded data
├── Dockerfile             # Contained deployment (the recommended way to run this)
├── packages.txt           # apt deps for Streamlit Community Cloud (libgomp1 -> LightGBM)
├── .streamlit/
│   ├── config.toml        # Server + theme settings
│   └── secrets.toml.example
├── .env.example
├── .dockerignore
├── requirements.txt
├── .gitignore
├── bootstrap.py           # Assembles downloads into a project; --check health-checks it
├── BUILD.md               # Zero to running: what you're building and how to assemble it
├── SETUP.md               # Step-by-step setup & deployment guide
└── README.md
```

Runtime artifacts land in `runs/<run_id>/plots/` and are gitignored.

---

## Extending it

- **New file type** — add a parser to `ingestion.py` and register its suffix in `ingest_bytes`.
- **New pipeline stage** — write a `_stage_x` generator in `AutoDSAgent` that delegates to
  `_execute_stage`, and add it to `run()`. Its cells flow into the notebook automatically.
- **Different modelling stack** — the stage instructions in `agent.py` are plain strings;
  swap LightGBM for XGBoost or PyTorch by editing the `_stage_model` prompt and `requirements.txt`.
- **Deeper planning** — `ANALYST_SYSTEM` defines the JSON contract; extend the schema and the
  new fields become available to every downstream prompt.

---

## Cost and runtime

A typical run is 5 API calls (plus one per repair). On a 10k-row, 20-column dataset expect
roughly 60–120 seconds end to end, dominated by model training rather than tokens.

---

## License

MIT — see `LICENSE`. Generated code and generated conclusions are unverified model output;
review both before any production decision.
