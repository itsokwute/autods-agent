"""
AutoDS-Agent :: Notebook Builder
================================

Serialises a completed agent run into a self-contained, executable Google Colab
notebook (nbformat v4.5).

The exported notebook is *reproducible without the app*: every ingested table is
embedded directly in the file as gzip+base64 CSV, so a reviewer can open the
notebook in Colab, hit "Run all", and land on the same numbers. Tables that are
too large to embed fall back to a clearly marked upload cell.

Only ``json`` and the standard library are required -- no nbformat dependency,
so the export path can never break on a notebook-schema version bump.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

# Per-table and total embedding budgets (compressed bytes).
MAX_EMBED_BYTES_PER_TABLE = 12 * 1024 * 1024
MAX_EMBED_BYTES_TOTAL = 30 * 1024 * 1024

PIP_PACKAGES = "pandas numpy scikit-learn lightgbm matplotlib seaborn"


# --------------------------------------------------------------------------- #
# Cell primitives
# --------------------------------------------------------------------------- #
def _cell_id() -> str:
    return uuid.uuid4().hex[:12]


def _lines(source: str) -> List[str]:
    """nbformat stores source as a list of lines, each keeping its newline."""
    text = source.rstrip("\n")
    if not text:
        return []
    parts = text.split("\n")
    return [line + "\n" for line in parts[:-1]] + [parts[-1]]


def markdown_cell(source: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "id": _cell_id(), "metadata": {}, "source": _lines(source)}


def code_cell(source: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _lines(source),
    }


# --------------------------------------------------------------------------- #
# Data embedding
# --------------------------------------------------------------------------- #
def _encode_frame(frame: pd.DataFrame) -> Optional[str]:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    payload = gzip.compress(buffer.getvalue().encode("utf-8"), compresslevel=9)
    if len(payload) > MAX_EMBED_BYTES_PER_TABLE:
        return None
    return base64.b64encode(payload).decode("ascii")


def _chunk(blob: str, width: int = 96) -> str:
    return "\n".join(f'    "{blob[i : i + width]}"' for i in range(0, len(blob), width))


def build_data_cells(frames: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    embedded: Dict[str, str] = {}
    skipped: List[str] = []
    budget = MAX_EMBED_BYTES_TOTAL

    for key, frame in frames.items():
        blob = _encode_frame(frame)
        if blob is None or len(blob) > budget:
            skipped.append(key)
            continue
        embedded[key] = blob
        budget -= len(blob)

    cells: List[Dict[str, Any]] = [
        markdown_cell(
            "## 1. Data\n\n"
            "The tables below are embedded in this notebook as gzip+base64 CSV, so the "
            "analysis reproduces end-to-end with no external files."
        )
    ]

    for key, blob in embedded.items():
        shape = frames[key].shape
        cells.append(
            code_cell(
                f"# Table: {key}  (rows={shape[0]:,}, columns={shape[1]})\n"
                f"_{key}_b64 = (\n{_chunk(blob)}\n)\n"
                f"dataframes['{key}'] = _load_embedded(_{key}_b64)\n"
                f"print('{key}:', dataframes['{key}'].shape)"
            )
        )

    if skipped:
        cells.append(
            markdown_cell(
                "### Tables too large to embed\n\n"
                f"`{', '.join(skipped)}` exceeded the embedding budget. Upload the original "
                "file(s) in the next cell before running the analysis."
            )
        )
        cells.append(
            code_cell(
                "# Colab upload fallback for the tables that could not be embedded.\n"
                "try:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    for _name, _payload in uploaded.items():\n"
                "        _key = _name.rsplit('.', 1)[0].lower().replace(' ', '_')\n"
                "        dataframes[_key] = pd.read_csv(io.BytesIO(_payload))\n"
                "        print(_key, dataframes[_key].shape)\n"
                "except ImportError:\n"
                "    print('Not running in Colab -- load the missing tables manually into `dataframes`.')"
            )
        )

    cells.append(
        code_cell(
            "# Primary table and document context\n"
            "PRIMARY_TABLE = max(dataframes, key=lambda k: dataframes[k].shape[0] * dataframes[k].shape[1])\n"
            "df = dataframes[PRIMARY_TABLE].copy()\n"
            "documents = {}\n"
            "print('Primary table:', PRIMARY_TABLE, df.shape)\n"
            "df.head()"
        )
    )
    return cells


# --------------------------------------------------------------------------- #
# Spec / report rendering
# --------------------------------------------------------------------------- #
def _spec_markdown(spec: Dict[str, Any]) -> str:
    if not spec:
        return "_No analysis specification was produced._"

    def fmt(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return ", ".join(f"`{v}`" for v in value) if value else "_none_"
        if value in (None, ""):
            return "_none_"
        return str(value)

    rows = [
        ("ML paradigm", fmt(spec.get("ml_paradigm"))),
        ("Primary table", fmt(spec.get("primary_table"))),
        ("Target variable", fmt(spec.get("target_variable"))),
        ("Evaluation metric", fmt(spec.get("evaluation_metric"))),
        ("Validation strategy", fmt(spec.get("validation_strategy"))),
        ("Confidence", fmt(spec.get("confidence"))),
    ]
    table = "| Field | Value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)

    sections = [
        "## 0. Problem Definition",
        "",
        str(spec.get("problem_statement", "")).strip(),
        "",
        table,
        "",
    ]
    if spec.get("target_rationale"):
        sections += [f"**Why this target.** {spec['target_rationale']}", ""]
    if spec.get("business_context"):
        sections += [f"**Business context.** {spec['business_context']}", ""]
    for label, key in (
        ("Feature engineering plan", "feature_engineering_plan"),
        ("Leakage risks", "leakage_risks"),
        ("Data quality issues", "data_quality_issues"),
    ):
        items = spec.get(key) or []
        if items:
            sections.append(f"**{label}**")
            sections += [f"- {item}" for item in items]
            sections.append("")
    return "\n".join(sections)


def _framing_markdown(framing: Dict[str, Any]) -> str:
    """Render the business questions and problems into the exported notebook."""
    if not framing:
        return ""
    icons = {"answerable": "OK", "partial": "PARTIAL", "not_answerable": "NOT POSSIBLE"}
    out = ["## Business Framing", ""]
    for table, block in framing.items():
        out.append(f"### `{table}`")
        if block.get("dataset_summary"):
            out += ["", block["dataset_summary"], ""]

        for label, key, shortfall_key, field in (
            ("Business questions", "business_questions", "questions_shortfall", "question"),
            ("Business problems", "business_problems", "problems_shortfall", "problem"),
        ):
            items = block.get(key) or []
            out.append(f"**{label}**")
            if block.get(shortfall_key):
                out += ["", f"> Only {len(items)} of 5. {block[shortfall_key]}", ""]
            if not items:
                out += ["", "_None supportable by this dataset._", ""]
                continue
            out.append("")
            out.append("| # | Item | Feasibility | Columns |")
            out.append("|---|---|---|---|")
            for i, item in enumerate(items, start=1):
                text = str(item.get(field, "")).replace("|", "/")
                feas = icons.get(str(item.get("feasibility", "")).lower(), "-")
                cols = ", ".join(f"`{c}`" for c in (item.get("columns_used") or [])[:5])
                out.append(f"| {i} | {text} | {feas} | {cols} |")
            out.append("")
    return "\n".join(out)


def _stage_heading(index: int, cell: Any) -> str:
    title = getattr(cell, "title", "Step")
    stage = getattr(cell, "stage", "")
    attempts = getattr(cell, "attempts", 1)
    caption = (getattr(cell, "caption", "") or "").strip()

    parts = [f"## {index}. {title}", ""]
    if caption:
        parts += [f"_{caption}_", ""]
    parts.append(f"<sub>Stage: `{stage}`</sub>")
    if attempts and attempts > 1:
        parts.append(
            f"\n> Self-corrected: this step needed {attempts} generation attempts before it ran cleanly."
        )
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Notebook assembly
# --------------------------------------------------------------------------- #
def build_notebook(
    frames: Dict[str, pd.DataFrame],
    bundle: Dict[str, Any],
    include_pip_cell: bool = True,
) -> Dict[str, Any]:
    """Assemble a complete nbformat v4.5 notebook dictionary."""
    spec = bundle.get("spec") or {}
    results = bundle.get("results") or {}
    report = bundle.get("report") or ""
    transcript: Sequence[Any] = bundle.get("transcript") or []
    run_id = bundle.get("run_id", "run")
    goal = bundle.get("user_goal") or "_none supplied_"
    generated = time.strftime("%Y-%m-%d %H:%M:%S")

    cells: List[Dict[str, Any]] = [
        markdown_cell(
            "# AutoDS-Agent — Automated Analysis Report\n\n"
            f"**Run ID:** `{run_id}`  \n"
            f"**Generated:** {generated}  \n"
            f"**Planner model:** `{bundle.get('model', 'n/a')}`  \n"
            f"**Random state:** `{bundle.get('random_state', 42)}`\n\n"
            f"**User goal:** {goal}\n\n"
            "This notebook is a faithful replay of an autonomous agent run: the same code "
            "that executed in the app, in the same order, against the same data. Run all "
            "cells top to bottom to reproduce every result."
        )
    ]

    if include_pip_cell:
        cells.append(
            code_cell(
                "# Colab bootstrap — safe to skip if the environment is already provisioned.\n"
                f"%pip install -q {PIP_PACKAGES}"
            )
        )

    cells.append(
        code_cell(
            "import base64, gzip, io, json, math, re, itertools, warnings\n"
            "warnings.filterwarnings('ignore')\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib\n"
            "import matplotlib.pyplot as plt\n"
            "try:\n"
            "    import seaborn as sns\n"
            "    sns.set_theme(style='whitegrid')\n"
            "except Exception:\n"
            "    sns = None\n"
            "\n"
            "pd.set_option('display.max_columns', 80)\n"
            "pd.set_option('display.width', 160)\n"
            f"RANDOM_STATE = {bundle.get('random_state', 42)}\n"
            "np.random.seed(RANDOM_STATE)\n"
            "RESULTS = {}\n"
            "dataframes = {}\n"
            "\n"
            "# Chart styling: every figure gets room, readable type, automatic layout.\n"
            "plt.rcParams.update({\n"
            "    'figure.figsize': (10, 6), 'figure.dpi': 110, 'figure.autolayout': True,\n"
            "    'figure.facecolor': 'white', 'axes.titlesize': 14, 'axes.titleweight': 'bold',\n"
            "    'axes.titlepad': 14, 'axes.labelsize': 11.5, 'axes.labelpad': 9,\n"
            "    'axes.grid': True, 'axes.axisbelow': True, 'grid.alpha': 0.3,\n"
            "    'legend.frameon': True, 'legend.framealpha': 0.9, 'legend.fontsize': 10,\n"
            "    'xtick.labelsize': 10, 'ytick.labelsize': 10,\n"
            "    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.35,\n"
            "})\n"
            "\n"
            "_CHART_META = {}\n"
            "\n"
            "def chart(title, xlabel='', ylabel='', caption='', figsize=(10, 6)):\n"
            '    """Create ONE properly labelled figure. Returns (fig, ax)."""\n'
            "    fig, ax = plt.subplots(figsize=figsize)\n"
            "    ax.set_title(title)\n"
            "    if xlabel:\n"
            "        ax.set_xlabel(xlabel)\n"
            "    if ylabel:\n"
            "        ax.set_ylabel(ylabel)\n"
            "    _CHART_META[fig.number] = {'title': str(title), 'caption': str(caption or title)}\n"
            "    return fig, ax\n"
            "\n"
            "def _load_embedded(blob):\n"
            "    \"\"\"Decode a gzip+base64 CSV embedded in this notebook.\"\"\"\n"
            "    raw = gzip.decompress(base64.b64decode(blob))\n"
            "    return pd.read_csv(io.BytesIO(raw))\n"
            "\n"
            "print('Environment ready.')"
        )
    )

    cells.extend(build_data_cells(frames))
    framing_md = _framing_markdown(bundle.get("framing") or {})
    if framing_md:
        cells.append(markdown_cell(framing_md))
    cells.append(markdown_cell(_spec_markdown(spec)))

    for i, cell in enumerate(transcript, start=2):
        cells.append(markdown_cell(_stage_heading(i, cell)))
        cells.append(code_cell(getattr(cell, "code", "")))

    if results:
        cells.append(markdown_cell("## Final Metrics\n\nThe `RESULTS` dictionary produced by the run."))
        cells.append(
            code_cell(
                "print(json.dumps(RESULTS, indent=2, default=str))"
                if transcript
                else f"RESULTS = {json.dumps(results, indent=2, default=str)}\nprint(json.dumps(RESULTS, indent=2))"
            )
        )
        cells.append(
            markdown_cell(
                "<details><summary>Metrics recorded during the original run</summary>\n\n"
                "```json\n" + json.dumps(results, indent=2, default=str)[:20_000] + "\n```\n\n</details>"
            )
        )

    cells.append(markdown_cell(report if report.strip() else "## Conclusion\n\n_No report was generated._"))

    failures = bundle.get("failures") or []
    if failures:
        rows = "\n".join(
            f"| {f.get('stage', '')} | {f.get('attempt', '')} | {str(f.get('error', ''))[:160]} |"
            for f in failures
        )
        cells.append(
            markdown_cell(
                "## Appendix — Self-Correction Log\n\n"
                "Errors the agent hit and repaired during the run. Kept for auditability.\n\n"
                "| Stage | Attempt | Error |\n|---|---|---|\n" + rows
            )
        )

    cells.append(
        markdown_cell(
            "---\n\n_Generated by **AutoDS-Agent**. Review the generated code before using "
            "any result in a production decision._"
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "autods": {"run_id": run_id, "generated": generated, "model": bundle.get("model")},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def notebook_to_bytes(notebook: Dict[str, Any]) -> bytes:
    return json.dumps(notebook, indent=1, ensure_ascii=False).encode("utf-8")


def suggested_filename(bundle: Dict[str, Any]) -> str:
    spec = bundle.get("spec") or {}
    base = str(spec.get("target_variable") or spec.get("ml_paradigm") or "analysis")
    base = re.sub(r"[^0-9A-Za-z]+", "_", base).strip("_").lower() or "analysis"
    return f"autods_{base}_{bundle.get('run_id', 'run')}.ipynb"


def export_notebook(
    frames: Dict[str, pd.DataFrame], bundle: Dict[str, Any], path: str | Path
) -> Path:
    """Write the notebook to disk and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(notebook_to_bytes(build_notebook(frames, bundle)))
    return target


__all__ = [
    "build_notebook",
    "export_notebook",
    "notebook_to_bytes",
    "suggested_filename",
]
