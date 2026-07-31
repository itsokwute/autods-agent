"""
AutoDS-Agent :: Agentic Backend
===============================

A plan -> generate -> execute -> repair loop driven by the Anthropic Messages
API and grounded in a *stateful* Python REPL.

Pipeline
--------
    1. PROFILE   deterministic schema/missingness profiling (no tokens burned)
    2. ANALYZE   Claude emits a strict-JSON AnalysisSpec: problem statement,
                 ML paradigm, target variable, feature typing, metric
    3. EDA       generated code, executed, plots captured
    4. PREPARE   cleaning, preprocessing, feature engineering -> X_train/X_test
    5. MODEL     scikit-learn baseline + LightGBM, metrics into RESULTS
    6. REPORT    Claude synthesises "Conclusion & Business Impact" + "Future Work"

Every stage runs through :meth:`AutoDSAgent._execute_stage`, which retries with
the full traceback fed back to the model until the cell runs clean or the
repair budget is exhausted. Everything that executed successfully is retained
in ``agent.transcript`` and replayed into the exported Colab notebook.

SECURITY
--------
The REPL executes model-generated code **in this process**. The static guard
below blocks the obvious footguns, but it is a speed bump, not a sandbox. For
untrusted input or multi-tenant deployment, run the app inside a container
with no credentials mounted and no outbound network. See README.md.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import textwrap
import threading
import time
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import pandas as pd

from ingestion import json_safe, profile_frames

LOGGER = logging.getLogger("autods.agent")

try:  # pragma: no cover - environment dependent
    import anthropic
except Exception:  # pragma: no cover
    anthropic = None


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
# NOTE: the original brief named "Claude 3.7 Sonnet". That snapshot is legacy;
# the default below is a current model. Any listed id can be selected in the UI.
DEFAULT_MODEL = "claude-sonnet-5"
SUPPORTED_MODELS = [
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-haiku-4-5-20251001",
    "claude-3-7-sonnet-latest",   # legacy, kept for parity with the original spec
]

STAGES = ["profile", "framing", "analyze", "eda", "charts", "prepare", "model", "report"]


@dataclass
class AgentConfig:
    api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model: str = DEFAULT_MODEL
    max_tokens: int = 8000
    # Left as None: current models reject `temperature`. Set a float only for
    # legacy models that still accept it.
    temperature: Optional[float] = None
    max_repair_attempts: int = 3
    sample_rows: int = 5
    execution_timeout: int = 600
    random_state: int = 42
    workdir: str = "runs"
    enable_lightgbm: bool = True
    n_charts: int = 5
    n_business_items: int = 5
    max_framing_tables: int = 4
    allow_network_in_repl: bool = False


@dataclass
class AgentEvent:
    """One item in the real-time execution stream rendered by the UI."""

    kind: str                      # status|spec|code|stdout|error|repair|plot|report|done
    stage: str
    content: str = ""
    data: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChartArtifact:
    """One saved figure, with the labelling needed to present it on its own."""

    path: str
    title: str
    caption: str = ""
    stage: str = ""


@dataclass
class ExecResult:
    ok: bool
    stdout: str = ""
    error: str = ""
    traceback: str = ""
    plots: List[ChartArtifact] = field(default_factory=list)
    elapsed: float = 0.0


@dataclass
class TranscriptCell:
    """A successfully executed step, replayed verbatim into the notebook."""

    stage: str
    title: str
    code: str
    stdout: str = ""
    plots: List[ChartArtifact] = field(default_factory=list)
    attempts: int = 1
    caption: str = ""


# --------------------------------------------------------------------------- #
# Stateful REPL
# --------------------------------------------------------------------------- #
_BANNED_PATTERNS = [
    (r"\bsubprocess\b", "subprocess is not permitted"),
    (r"\bos\.system\b", "os.system is not permitted"),
    (r"\bos\.popen\b", "os.popen is not permitted"),
    (r"\bshutil\.rmtree\b", "recursive delete is not permitted"),
    (r"\bos\.remove\b", "file deletion is not permitted"),
    (r"\bsys\.exit\b", "sys.exit is not permitted"),
    (r"\b__import__\s*\(", "__import__ is not permitted"),
    (r"\beval\s*\(", "eval is not permitted"),
    (r"\bexec\s*\(", "exec is not permitted"),
    (r"\bpip\s+install\b", "runtime installs are not permitted"),
]
_NETWORK_PATTERNS = [
    (r"\brequests\.(get|post|put|delete)\b", "network access is disabled"),
    (r"\burllib\.request\b", "network access is disabled"),
    (r"\bsocket\.", "network access is disabled"),
]

_REPL_PREAMBLE = """
import warnings, json, math, re, itertools
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_theme(style='whitegrid')
except Exception:
    sns = None

# Every figure gets room to breathe, readable type, and automatic layout.
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 110,
    'figure.autolayout': True,
    'figure.facecolor': 'white',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.titlepad': 14,
    'axes.labelsize': 11.5,
    'axes.labelpad': 9,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.alpha': 0.3,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.35,
})

_CHART_META = {}

def chart(title, xlabel='', ylabel='', caption='', figsize=(10, 6)):
    \"\"\"Create ONE properly labelled figure. Returns (fig, ax).

    Always use this instead of plt.subplots() so the chart is titled, labelled,
    sized, and captioned for the report.
    \"\"\"
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    _CHART_META[fig.number] = {'title': str(title), 'caption': str(caption or title)}
    return fig, ax

pd.set_option('display.max_columns', 80)
pd.set_option('display.width', 160)
RESULTS = {}
"""


class PythonREPL:
    """A persistent namespace: variables survive across stages, like a notebook."""

    def __init__(self, plot_dir: Path, allow_network: bool = False) -> None:
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.allow_network = allow_network
        self.namespace: Dict[str, Any] = {"__name__": "__autods__"}
        self._plot_counter = 0
        exec(compile(_REPL_PREAMBLE, "<preamble>", "exec"), self.namespace)

    # -- state ------------------------------------------------------------- #
    def inject(self, frames: Dict[str, pd.DataFrame], texts: Optional[Dict[str, str]] = None) -> None:
        self.namespace["dataframes"] = {k: v.copy() for k, v in frames.items()}
        self.namespace["documents"] = dict(texts or {})
        if frames:
            primary = max(frames.items(), key=lambda kv: kv[1].shape[0] * kv[1].shape[1])[0]
            self.namespace["df"] = frames[primary].copy()
            self.namespace["PRIMARY_TABLE"] = primary

    def get(self, name: str, default: Any = None) -> Any:
        return self.namespace.get(name, default)

    def variables(self) -> Dict[str, str]:
        import types

        skip = {"__builtins__", "__name__"}
        out: Dict[str, str] = {}
        for key, value in self.namespace.items():
            if key in skip or key.startswith("_") or callable(value) or isinstance(value, type):
                continue
            if isinstance(value, types.ModuleType):
                continue
            if isinstance(value, pd.DataFrame):
                out[key] = f"DataFrame{value.shape}"
            elif isinstance(value, pd.Series):
                out[key] = f"Series(len={len(value)})"
            elif isinstance(getattr(value, "shape", None), tuple):
                out[key] = f"{type(value).__name__}{value.shape}"
            elif isinstance(value, (int, float, str, bool)):
                out[key] = f"{type(value).__name__}={str(value)[:40]}"
            elif isinstance(value, (dict, list, set, tuple)):
                out[key] = f"{type(value).__name__}(len={len(value)})"
            else:
                out[key] = type(value).__name__
        return out

    # -- guards ------------------------------------------------------------ #
    def screen(self, code: str) -> Optional[str]:
        patterns = list(_BANNED_PATTERNS)
        if not self.allow_network:
            patterns += _NETWORK_PATTERNS
        for pattern, reason in patterns:
            if re.search(pattern, code):
                return reason
        return None

    # -- execution --------------------------------------------------------- #
    def _harvest_plots(self, stage: str) -> List[ChartArtifact]:
        import matplotlib.pyplot as plt

        saved: List[ChartArtifact] = []
        meta = self.namespace.get("_CHART_META", {})
        for num in plt.get_fignums():
            fig = plt.figure(num)
            axes = fig.get_axes()
            if not axes:
                plt.close(fig)
                continue
            self._plot_counter += 1
            info = meta.get(num, {}) if isinstance(meta, dict) else {}
            suptitle = getattr(getattr(fig, "_suptitle", None), "get_text", lambda: "")()
            title = (
                info.get("title")
                or suptitle
                or axes[0].get_title()
                or f"Figure {self._plot_counter}"
            )
            path = self.plot_dir / f"{stage}_{self._plot_counter:02d}.png"
            try:
                fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
                saved.append(
                    ChartArtifact(
                        path=str(path),
                        title=str(title).strip(),
                        caption=str(info.get("caption") or title).strip(),
                        stage=stage,
                    )
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Could not save figure: %s", exc)
            finally:
                plt.close(fig)
        if isinstance(meta, dict):
            meta.clear()   # figure numbers are reused; don't leak captions across cells
        return saved

    def run(self, code: str, stage: str = "cell", timeout: Optional[int] = None) -> ExecResult:
        violation = self.screen(code)
        if violation:
            return ExecResult(
                ok=False,
                error=f"BlockedOperation: {violation}",
                traceback=f"BlockedOperation: {violation}. Rewrite the cell without it.",
            )

        buffer = io.StringIO()
        holder: Dict[str, Any] = {}
        started = time.time()

        def _target() -> None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with redirect_stdout(buffer), redirect_stderr(buffer):
                        exec(compile(code, f"<{stage}>", "exec"), self.namespace)
                holder["ok"] = True
            except BaseException as exc:  # noqa: BLE001 - we surface everything to the model
                holder["ok"] = False
                holder["error"] = f"{type(exc).__name__}: {exc}"
                holder["traceback"] = _trim_traceback(traceback.format_exc())

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout or 10**6)

        if thread.is_alive():
            # Python cannot safely kill a running thread; we report and move on.
            return ExecResult(
                ok=False,
                stdout=_tail(buffer.getvalue()),
                error="TimeoutError: cell exceeded the execution budget",
                traceback=(
                    "TimeoutError: the cell exceeded the execution budget. "
                    "Rewrite it to be cheaper -- subsample rows, cut n_estimators, "
                    "reduce cross-validation folds."
                ),
                elapsed=time.time() - started,
            )

        return ExecResult(
            ok=bool(holder.get("ok")),
            stdout=_tail(buffer.getvalue()),
            error=holder.get("error", ""),
            traceback=holder.get("traceback", ""),
            plots=self._harvest_plots(stage),
            elapsed=time.time() - started,
        )


def _tail(text: str, limit: int = 12_000) -> str:
    text = text.rstrip()
    return text if len(text) <= limit else "... [output truncated]\n" + text[-limit:]


def _trim_traceback(tb: str, limit: int = 4_000) -> str:
    lines = [ln for ln in tb.splitlines() if "site-packages" not in ln or "Error" in ln]
    trimmed = "\n".join(lines)
    return trimmed if len(trimmed) <= limit else trimmed[-limit:]


# --------------------------------------------------------------------------- #
# Anthropic client wrapper
# --------------------------------------------------------------------------- #
class ClaudeClient:
    """Thin wrapper: retries, JSON coercion, and code-fence stripping."""

    def __init__(self, config: AgentConfig) -> None:
        if anthropic is None:
            raise RuntimeError("The 'anthropic' package is not installed. pip install anthropic")
        if not config.api_key:
            raise RuntimeError("No Anthropic API key. Set ANTHROPIC_API_KEY or paste one in the sidebar.")
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.api_key)
        self.input_tokens = 0
        self.output_tokens = 0
        # Newer models reject `temperature` outright. Only send it when the user has
        # explicitly asked for one, and drop it permanently if the API objects.
        self._send_temperature = config.temperature is not None

    def complete(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        retries: int = 4,
    ) -> str:
        last_error: Optional[Exception] = None
        attempt = 0
        while attempt < retries:
            payload: Dict[str, Any] = {
                "model": self.config.model,
                "max_tokens": max_tokens or self.config.max_tokens,
                "system": system,
                "messages": messages,
            }
            chosen = self.config.temperature if temperature is None else temperature
            if self._send_temperature and chosen is not None:
                payload["temperature"] = chosen

            try:
                response = self.client.messages.create(**payload)
                usage = getattr(response, "usage", None)
                if usage is not None:
                    self.input_tokens += getattr(usage, "input_tokens", 0) or 0
                    self.output_tokens += getattr(usage, "output_tokens", 0) or 0
                return "".join(
                    block.text for block in response.content if getattr(block, "type", "") == "text"
                ).strip()
            except Exception as exc:
                message = str(exc).lower()
                # Self-heal: the model does not accept this parameter. Drop it and
                # retry immediately without burning a retry.
                if "temperature" in message and self._send_temperature:
                    self._send_temperature = False
                    LOGGER.info("Model rejected `temperature`; retrying without it.")
                    continue
                attempt += 1
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(min(2 ** (attempt - 1) * 1.5, 20))
        raise RuntimeError(f"Anthropic API call failed after {retries} attempts: {last_error}")

    def complete_json(self, system: str, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        raw = self.complete(system, messages, **kwargs)
        parsed = extract_json(raw)
        if parsed is not None:
            return parsed
        repair = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "That was not valid JSON. Reply with the JSON object only -- no prose, no code fences."},
        ]
        raw2 = self.complete(system, repair, **kwargs)
        parsed = extract_json(raw2)
        if parsed is None:
            raise ValueError(f"Model did not return parseable JSON.\n---\n{raw2[:1500]}")
        return parsed


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except Exception:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except Exception:
            return None
    return None


def extract_code(text: str) -> str:
    """Pull the python cell out of a model response."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
ANALYST_SYSTEM = """You are a Lead Data Scientist scoping a new engagement.

You receive machine-generated profiles of one or more pandas DataFrames: dtypes,
missingness, cardinality, summary statistics and sample rows, plus any free text
extracted from accompanying documents.

Return ONE JSON object and nothing else. Schema:

{
  "problem_statement": "2-3 sentences: what question this data can answer and why it matters",
  "primary_table": "key of the table to model",
  "ml_paradigm": "binary_classification | multiclass_classification | regression | time_series_forecasting | clustering | anomaly_detection | exploratory_only",
  "target_variable": "column name, or null for unsupervised/exploratory",
  "target_rationale": "why this target (or why none is appropriate)",
  "id_columns": ["columns that are identifiers and must be dropped before modelling"],
  "datetime_columns": ["..."],
  "numeric_features": ["..."],
  "categorical_features": ["..."],
  "text_features": ["free-text columns needing vectorisation"],
  "leakage_risks": ["columns that would leak the target, with a one-line reason each"],
  "data_quality_issues": ["concrete issues visible in the profile"],
  "evaluation_metric": "primary metric, e.g. roc_auc / f1_macro / rmse / mae / silhouette",
  "validation_strategy": "e.g. stratified 5-fold CV, time-ordered split with a 20% holdout",
  "class_balance_note": "for classification: observed balance and how to handle it; else null",
  "feature_engineering_plan": ["3-6 concrete, dataset-specific ideas"],
  "business_context": "who consumes this and what decision it changes",
  "confidence": "high | medium | low"
}

Rules:
- Choose the target only from columns that actually exist in the primary table.
- If nothing is predictable, say so honestly with ml_paradigm "exploratory_only" and target null.
- Never invent column names. Never wrap the JSON in prose or code fences."""


CODER_SYSTEM = """You are a Principal Data Scientist writing code for a persistent Python session.

ENVIRONMENT
- The session is stateful: every variable you create survives into later stages.
- Already imported: numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
  (sns may be None), plus warnings/json/math/re/itertools. RESULTS = {} exists.
- `dataframes` is a dict[str, DataFrame] of every ingested table. `df` is the primary
  table. `documents` is a dict[str, str] of extracted document text.
- scikit-learn is available. lightgbm may be available -- always guard it:
      try:
          import lightgbm as lgb
          HAS_LGB = True
      except Exception:
          HAS_LGB = False

HARD RULES
1. Reply with exactly ONE ```python code block and no prose outside it.
2. No network calls, no shell commands, no file deletion, no pip installs, no exec/eval.
3. Do NOT re-read data from disk. Use the in-memory objects.
4. CHARTS -- these rules are strict, because the charts are the deliverable:
   - Build every figure with the provided helper:
         fig, ax = chart('Title', xlabel='X (units)', ylabel='Y (units)',
                         caption='One sentence on what this shows and what to look for.')
   - ONE chart per figure. Never plt.subplots(2, 2). Never cram panels together.
     If you want four views, make four separate figures.
   - Every chart needs a specific title (not 'Distribution'), axis labels with units
     where known, and a legend whenever more than one series is drawn --
     ax.legend(title='...') with readable labels, never the raw column name.
   - Rotate tick labels (ax.tick_params(axis='x', rotation=45)) when they would collide,
     and cap categorical charts at the top 15 categories.
   - Do NOT call plt.show(), plt.tight_layout(), or close anything -- all handled.
5. print() every result you want the report to be able to cite. Silent cells are useless.
6. Write defensively: check a column exists before using it, guard empty selections,
   subsample if the data is large (>200k rows) and say so in a printed note.
7. The code must be idempotent and safe to re-run top to bottom in a fresh notebook.
8. Keep runtime under a couple of minutes. Prefer n_estimators<=400, cv<=5.
9. Use random_state=RANDOM_STATE everywhere a seed is accepted."""


BUSINESS_FRAMING_SYSTEM = """You are a Lead Data Scientist in a first meeting with a business owner.
You have their data and nothing else. Your job is to tell them what this data could be used
for -- before anyone has decided what to model.

You receive a machine-generated profile of ONE table: dtypes, missingness, cardinality,
summary statistics and sample rows, plus any document text that came with it.

Return ONE JSON object and nothing else:

{
  "dataset_summary": "one sentence: what this data appears to record, and at what grain (one row = ?)",
  "domain_guess": "the business domain this most likely comes from",
  "business_questions": [
    {
      "question": "a question a manager would actually ask, phrased in their words, not statistical language",
      "why_it_matters": "the decision or cost this bears on, one sentence",
      "columns_used": ["the columns that answer it"],
      "analysis_needed": "the specific analysis that would answer it",
      "feasibility": "answerable | partial | not_answerable",
      "feasibility_note": "for 'partial', the assumption required; for 'not_answerable', the exact data that is missing"
    }
  ],
  "business_problems": [
    {
      "problem": "a concrete operational problem this data could help solve",
      "who_owns_it": "the role that owns this problem",
      "decision_it_changes": "what someone would do differently",
      "how_to_measure_improvement": "the metric that would move, and roughly by how much it would need to move to be worth doing",
      "columns_used": ["..."],
      "feasibility": "answerable | partial | not_answerable",
      "feasibility_note": "..."
    }
  ],
  "questions_shortfall": null,
  "problems_shortfall": null
}

Rules:
- Aim for FIVE questions and FIVE problems.
- HONESTY OVER COMPLETENESS. If this table genuinely cannot support five of either, return
  only the ones that hold up and put a plain-English explanation in "questions_shortfall"
  or "problems_shortfall" -- e.g. "Only 3 questions: the table has no time dimension and no
  outcome column, so nothing forward-looking or causal can be asked of it."
  Never invent a question to reach five. A padded list is a worse failure than a short one.
- Set the shortfall field to null when you return five.
- Questions and problems must be different things. A question is something to find out; a
  problem is something to fix. Do not restate one as the other.
- Every item must name real columns from the profile. Never invent a column.
- Be specific to THIS data. "Can we improve efficiency?" is worthless. "Which of the 4
  departments loses staff fastest in their first 12 months?" is useful.
- Rank both lists by business value, highest first."""


CHART_PLANNER_SYSTEM = """You are a Lead Data Scientist deciding which charts belong in a report.

You receive dataframe profiles and the analysis specification. Choose the charts that would
actually change a reader's understanding -- not a mechanical sweep of every column.

Return ONE JSON object and nothing else:

{
  "charts": [
    {
      "title": "specific, informative chart title naming the columns involved",
      "chart_type": "histogram | bar | box | scatter | line | heatmap | count",
      "columns": ["the columns this chart uses"],
      "why": "the question this chart answers, one sentence",
      "caption": "what the reader should look for, one sentence"
    }
  ],
  "skipped": ["charts you considered and rejected, with a short reason each"]
}

Rules:
- Use only columns that exist in the profile. Never invent one.
- Each chart must answer a different question. No near-duplicates.
- If a target variable exists, at least two charts must relate features to it.
- Order them so a reader moving top to bottom builds understanding.
- If the data supports fewer good charts than requested, return fewer and say why in
  "skipped". Do not pad."""


REPORTER_SYSTEM = """You are a Lead Data Scientist writing the closing section of a client report.

You receive the analysis specification, the printed output of every executed stage, and
the final metrics dictionary. Write in Markdown, grounded strictly in what the logs show.
Never invent a number that does not appear in the evidence.

Use exactly this structure:

## Conclusion & Business Impact
Three to five short paragraphs. Lead with the headline result and its metric. State what
the model is good enough for and what it is not. Quantify the impact in the operating
terms of whoever uses this (cost, throughput, yield, risk, hours saved), flagging clearly
where that translation is an assumption rather than a measurement.

### Key Findings
Four to six bullets, each anchored to a specific number from the logs.

### Limitations
Three to five bullets: sample size, leakage risk, drift, validation weaknesses, anything
the data simply cannot answer.

## Future Work
Five to eight bullets, ordered by expected value per unit of effort. Each names a concrete
next action and the expected gain. Include at least one data-collection item and one
deployment/monitoring item.

Be direct. No filler, no congratulation, no restating the brief."""


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #
class AutoDSAgent:
    def __init__(
        self,
        config: AgentConfig,
        frames: Dict[str, pd.DataFrame],
        texts: Optional[Dict[str, str]] = None,
        user_goal: str = "",
        target_override: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        if not frames:
            raise ValueError("AutoDSAgent needs at least one DataFrame.")
        self.config = config
        self.frames = frames
        self.texts = texts or {}
        self.user_goal = (user_goal or "").strip()
        self.target_override = target_override
        self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")

        self.run_dir = Path(config.workdir) / self.run_id
        self.plot_dir = self.run_dir / "plots"
        self.repl = PythonREPL(self.plot_dir, allow_network=config.allow_network_in_repl)
        self.repl.inject(frames, self.texts)
        self.repl.namespace["RANDOM_STATE"] = config.random_state

        self.client: Optional[ClaudeClient] = None
        self.profiles: Dict[str, Any] = {}
        self.spec: Dict[str, Any] = {}
        self.report: str = ""
        self.chart_plan: Dict[str, Any] = {}
        self.framing: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.transcript: List[TranscriptCell] = []
        self.plots: List[ChartArtifact] = []
        self.failures: List[Dict[str, str]] = []
        self.stage_logs: Dict[str, str] = {}

    # -- helpers ----------------------------------------------------------- #
    def _profile_json(self, limit: int = 30_000) -> str:
        blob = json.dumps(json_safe(self.profiles), indent=2, default=str)
        return blob if len(blob) <= limit else blob[:limit] + "\n... [profile truncated]"

    def _document_context(self, limit: int = 4_000) -> str:
        if not self.texts:
            return "(no document text supplied)"
        chunks = [f"### {name}\n{body[:1500]}" for name, body in list(self.texts.items())[:4]]
        joined = "\n\n".join(chunks)
        return joined[:limit]

    def _state_summary(self) -> str:
        variables = self.repl.variables()
        return "\n".join(f"  {k}: {v}" for k, v in list(variables.items())[:60]) or "  (empty)"

    def _tail_logs(self, limit: int = 6_000) -> str:
        joined = "\n\n".join(f"[{stage}]\n{log}" for stage, log in self.stage_logs.items())
        return joined if len(joined) <= limit else joined[-limit:]

    # -- core loop --------------------------------------------------------- #
    def _execute_stage(
        self, stage: str, title: str, instruction: str, caption: str = ""
    ) -> Iterator[AgentEvent]:
        """Generate -> execute -> repair until the cell runs clean or budget is out."""
        messages: List[Dict[str, Any]] = [{"role": "user", "content": instruction}]
        attempts = 0

        while attempts <= self.config.max_repair_attempts:
            attempts += 1
            yield AgentEvent("status", stage, f"Generating {title} code (attempt {attempts})...")
            raw = self.client.complete(CODER_SYSTEM, messages)  # type: ignore[union-attr]
            code = extract_code(raw)
            yield AgentEvent("code", stage, code, {"attempt": attempts})

            result = self.repl.run(code, stage=stage, timeout=self.config.execution_timeout)

            if result.stdout:
                yield AgentEvent("stdout", stage, result.stdout)
            for artifact in result.plots:
                self.plots.append(artifact)
                yield AgentEvent("plot", stage, artifact.title, {"chart": artifact})

            if result.ok:
                self.transcript.append(
                    TranscriptCell(
                        stage=stage,
                        title=title,
                        code=code,
                        stdout=result.stdout,
                        plots=result.plots,
                        attempts=attempts,
                        caption=caption,
                    )
                )
                self.stage_logs[stage] = result.stdout
                yield AgentEvent(
                    "status", stage, f"{title} completed in {result.elapsed:.1f}s.", {"ok": True}
                )
                return

            self.failures.append({"stage": stage, "attempt": str(attempts), "error": result.error})
            yield AgentEvent("error", stage, result.error, {"attempt": attempts})

            if attempts > self.config.max_repair_attempts:
                break

            yield AgentEvent("repair", stage, f"Self-correcting {title} (attempt {attempts + 1})...")
            messages += [
                {"role": "assistant", "content": f"```python\n{code}\n```"},
                {
                    "role": "user",
                    "content": (
                        "The cell failed.\n\n"
                        f"TRACEBACK\n{result.traceback}\n\n"
                        f"PARTIAL STDOUT\n{result.stdout[-1500:] or '(none)'}\n\n"
                        f"LIVE SESSION VARIABLES\n{self._state_summary()}\n\n"
                        "Diagnose the root cause and return the COMPLETE corrected cell. "
                        "Assume any object created before the failure may be in a half-built "
                        "state -- rebuild what you need rather than assuming it exists. "
                        "Reply with one ```python block only."
                    ),
                },
            ]

        yield AgentEvent(
            "status",
            stage,
            f"{title} abandoned after {attempts} attempts; continuing with the remaining stages.",
            {"ok": False},
        )

    # -- stages ------------------------------------------------------------ #
    def _stage_profile(self) -> Iterator[AgentEvent]:
        yield AgentEvent("status", "profile", "Profiling schemas, dtypes and missingness...")
        self.profiles = profile_frames(self.frames, sample_rows=self.config.sample_rows)
        shapes = ", ".join(f"{k}{v.shape}" for k, v in self.frames.items())
        yield AgentEvent("stdout", "profile", f"Ingested {len(self.frames)} table(s): {shapes}")

    def _stage_framing(self) -> Iterator[AgentEvent]:
        """Ask what this data is good for, before deciding what to model."""
        tables = sorted(
            self.frames.items(), key=lambda kv: kv[1].shape[0] * kv[1].shape[1], reverse=True
        )
        capped = tables[: self.config.max_framing_tables]
        if len(tables) > len(capped):
            yield AgentEvent(
                "status", "framing",
                f"{len(tables)} tables ingested; framing the {len(capped)} largest.",
            )

        wanted = self.config.n_business_items
        for key, _frame in capped:
            yield AgentEvent("status", "framing", f"Reading '{key}' for business questions and problems...")
            prompt = textwrap.dedent(
                f"""
                TABLE NAME
                {key}

                PROFILE
                {json.dumps(json_safe(self.profiles.get(key, {})), indent=2, default=str)[:20000]}

                DOCUMENT CONTEXT
                {self._document_context()}

                USER GOAL
                {self.user_goal or '(none supplied -- judge the data on its own merits)'}

                Return {wanted} business questions and {wanted} business problems, or fewer
                with an honest shortfall explanation.
                """
            ).strip()
            try:
                framing = self.client.complete_json(  # type: ignore[union-attr]
                    BUSINESS_FRAMING_SYSTEM, [{"role": "user", "content": prompt}]
                )
            except Exception as exc:
                yield AgentEvent("error", "framing", f"Could not frame '{key}': {exc}")
                continue

            questions = framing.get("business_questions") or []
            problems = framing.get("business_problems") or []
            self.framing[key] = framing
            yield AgentEvent("framing", "framing", key, {"table": key, "framing": framing})

            summary = f"'{key}': {len(questions)} question(s), {len(problems)} problem(s)."
            if len(questions) < wanted and framing.get("questions_shortfall"):
                summary += f" Fewer than {wanted} questions -- {framing['questions_shortfall']}"
            if len(problems) < wanted and framing.get("problems_shortfall"):
                summary += f" Fewer than {wanted} problems -- {framing['problems_shortfall']}"
            yield AgentEvent("status", "framing", summary)

    def _framing_digest(self, limit: int = 6000) -> str:
        """Compact view of the framing, for downstream prompts."""
        if not self.framing:
            return "(no business framing produced)"
        chunks = []
        for table, framing in self.framing.items():
            questions = [q.get("question", "") for q in (framing.get("business_questions") or [])]
            problems = [p.get("problem", "") for p in (framing.get("business_problems") or [])]
            chunks.append(
                f"### {table}\n{framing.get('dataset_summary', '')}\n"
                + "Questions:\n" + "\n".join(f"  - {q}" for q in questions)
                + "\nProblems:\n" + "\n".join(f"  - {p}" for p in problems)
            )
        joined = "\n\n".join(chunks)
        return joined if len(joined) <= limit else joined[:limit] + "\n... [truncated]"

    def _stage_analyze(self) -> Iterator[AgentEvent]:
        yield AgentEvent("status", "analyze", "Defining the problem and selecting a target...")
        goal = self.user_goal or "(none supplied -- infer the most valuable problem from the data)"
        override = (
            f"\nHARD CONSTRAINT: the user has forced the target variable to '{self.target_override}'. "
            "Use it and build the specification around it."
            if self.target_override
            else ""
        )
        prompt = (
            f"USER GOAL\n{goal}{override}\n\n"
            f"BUSINESS FRAMING ALREADY PRODUCED\n{self._framing_digest()}\n\n"
            f"DATAFRAME PROFILES\n{self._profile_json()}\n\n"
            f"DOCUMENT CONTEXT\n{self._document_context()}\n\n"
            "Return the specification JSON."
        )
        spec = self.client.complete_json(ANALYST_SYSTEM, [{"role": "user", "content": prompt}])  # type: ignore[union-attr]
        if self.target_override:
            spec["target_variable"] = self.target_override
        spec.setdefault("primary_table", self.repl.get("PRIMARY_TABLE"))
        self.spec = spec
        self.repl.namespace["SPEC"] = spec
        yield AgentEvent("spec", "analyze", json.dumps(spec, indent=2), {"spec": spec})

    def _stage_eda(self) -> Iterator[AgentEvent]:
        instruction = textwrap.dedent(
            f"""
            STAGE 1 -- EXPLORATORY ANALYSIS (STATISTICS ONLY -- NO CHARTS IN THIS CELL)

            ANALYSIS SPECIFICATION
            {json.dumps(json_safe(self.spec), indent=2)}

            DATAFRAME PROFILES
            {self._profile_json(18_000)}

            Charts are built in a later, dedicated stage. Do not create any figure here.

            Write one cell that:
              - selects the primary table into `df` (re-select from `dataframes` if the spec
                names a different table) and prints its shape;
              - prints dtypes, a missingness table (count and %), and describe() for numeric
                and for categorical columns;
              - reports duplicate rows and constant columns;
              - if a target is defined, prints its distribution and the strongest
                correlations or group differences against it, ranked;
              - ends by printing a short bulleted list of the concrete findings that should
                drive preprocessing decisions.
            """
        ).strip()
        yield from self._execute_stage("eda", "Exploratory Data Analysis", instruction)

    def _stage_charts(self) -> Iterator[AgentEvent]:
        """Plan a set of charts, then execute each one as its own cell."""
        yield AgentEvent("status", "charts", "Choosing which charts are worth drawing...")
        prompt = textwrap.dedent(
            f"""
            ANALYSIS SPECIFICATION
            {json.dumps(json_safe(self.spec), indent=2)}

            DATAFRAME PROFILES
            {self._profile_json(14_000)}

            EXPLORATORY FINDINGS
            {self.stage_logs.get('eda', '(none)')[-4000:]}

            Choose at most {self.config.n_charts} charts.
            """
        ).strip()

        try:
            plan = self.client.complete_json(  # type: ignore[union-attr]
                CHART_PLANNER_SYSTEM, [{"role": "user", "content": prompt}]
            )
        except Exception as exc:
            yield AgentEvent("error", "charts", f"Could not plan charts: {exc}")
            return

        specs = (plan.get("charts") or [])[: self.config.n_charts]
        self.chart_plan = plan
        if not specs:
            yield AgentEvent("status", "charts", "No charts were worth drawing for this data.")
            return

        skipped = plan.get("skipped") or []
        note = f" ({len(skipped)} rejected)" if skipped else ""
        yield AgentEvent("status", "charts", f"Drawing {len(specs)} chart(s){note}, one per cell.")

        for index, spec in enumerate(specs, start=1):
            title = str(spec.get("title") or f"Chart {index}")
            caption = str(spec.get("caption") or "")
            instruction = textwrap.dedent(
                f"""
                STAGE 2 -- CHART {index} OF {len(specs)}

                Build exactly ONE figure. Nothing else. No other charts in this cell.

                CHART SPECIFICATION
                {json.dumps(json_safe(spec), indent=2)}

                LIVE SESSION VARIABLES
                {self._state_summary()}

                Requirements:
                  - open with `fig, ax = chart('{title}', xlabel=..., ylabel=..., caption=...)`
                  - draw only what this specification asks for
                  - label the axes with units where they are known
                  - add ax.legend(title=...) with readable labels if more than one series
                  - sort bars/categories meaningfully and cap at the top 15
                  - rotate x tick labels if they would overlap
                  - print one line stating the single most important thing the chart shows
                """
            ).strip()
            yield from self._execute_stage("charts", f"Chart {index}: {title}", instruction, caption)

    def _stage_prepare(self) -> Iterator[AgentEvent]:
        instruction = textwrap.dedent(
            f"""
            STAGE 2 -- PREPROCESSING & FEATURE ENGINEERING

            ANALYSIS SPECIFICATION
            {json.dumps(json_safe(self.spec), indent=2)}

            EDA OUTPUT FROM THE PREVIOUS CELL
            {self.stage_logs.get('eda', '(EDA produced no output)')[-6000:]}

            LIVE SESSION VARIABLES
            {self._state_summary()}

            Write one cell that:
              - drops the id_columns and any leakage_risks named in the specification;
              - handles missing values explicitly (state the strategy per column type in a
                printed note) and de-duplicates rows;
              - engineers the features from feature_engineering_plan that the data supports,
                skipping any that are not feasible and printing why;
              - encodes categoricals and scales numerics using a sklearn ColumnTransformer
                assembled inside a Pipeline -- fit on train only, never on the full dataset;
              - for supervised work, produces `X_train, X_test, y_train, y_test` with the
                validation_strategy from the spec (stratify for classification, time-ordered
                split when datetime_columns drive the problem);
              - for unsupervised work, produces a single fitted feature matrix `X`;
              - stores the fitted transformer as `preprocessor` and the final feature name
                list as `feature_names`;
              - prints the resulting shapes and the first few engineered feature names.

            Do not fit any model in this cell.
            """
        ).strip()
        yield from self._execute_stage("prepare", "Preprocessing & Feature Engineering", instruction)

    def _stage_model(self) -> Iterator[AgentEvent]:
        lgb_note = (
            "Train BOTH a scikit-learn baseline and a LightGBM model, then compare them."
            if self.config.enable_lightgbm
            else "Train scikit-learn models only; LightGBM is disabled for this run."
        )
        instruction = textwrap.dedent(
            f"""
            STAGE 3 -- MODEL TRAINING & EVALUATION

            ANALYSIS SPECIFICATION
            {json.dumps(json_safe(self.spec), indent=2)}

            PREPARATION OUTPUT FROM THE PREVIOUS CELL
            {self.stage_logs.get('prepare', '(preparation produced no output)')[-6000:]}

            LIVE SESSION VARIABLES
            {self._state_summary()}

            {lgb_note}

            Write one cell that:
              - trains a simple, honest baseline first (DummyClassifier/DummyRegressor, plus
                a LogisticRegression/Ridge or RandomForest reference);
              - trains the gradient-boosted model with early stopping or modest n_estimators;
              - evaluates every model on the held-out set with the spec's evaluation_metric
                plus the standard supporting metrics for the paradigm (classification:
                accuracy, precision, recall, f1, ROC-AUC, confusion matrix; regression:
                RMSE, MAE, R2; clustering: silhouette and cluster sizes);
              - runs cross-validation on the winner and prints mean +/- std;
              - draws its diagnostic charts as SEPARATE figures, each via the `chart()`
                helper with a title, axis labels and a legend: model comparison, feature
                importance (or coefficients) for the winner, and ROC/PR curve for
                classification or predicted-vs-actual plus residuals for regression.
                One chart per figure -- never a grid of subplots;
              - writes a flat, JSON-serialisable summary into RESULTS, e.g.
                    RESULTS['paradigm'] = ...
                    RESULTS['target'] = ...
                    RESULTS['metric'] = ...
                    RESULTS['models'] = {{'model_name': {{'roc_auc': 0.87, ...}}, ...}}
                    RESULTS['best_model'] = ...
                    RESULTS['cv'] = {{'mean': ..., 'std': ..., 'folds': ...}}
                    RESULTS['top_features'] = [('feature', 0.31), ...]
              - prints RESULTS as formatted JSON at the end.

            If the paradigm is exploratory_only, instead build the strongest unsupervised
            characterisation the data supports (clustering with a silhouette sweep, or an
            anomaly score) and populate RESULTS accordingly.
            """
        ).strip()
        yield from self._execute_stage("model", "Model Training & Evaluation", instruction)
        raw_results = self.repl.get("RESULTS", {})
        if isinstance(raw_results, dict):
            self.results = json_safe(raw_results)

    def _stage_report(self) -> Iterator[AgentEvent]:
        yield AgentEvent("status", "report", "Synthesising the conclusion and future work...")
        prompt = textwrap.dedent(
            f"""
            BUSINESS FRAMING
            {self._framing_digest(4000)}

            ANALYSIS SPECIFICATION
            {json.dumps(json_safe(self.spec), indent=2)}

            FINAL METRICS (RESULTS)
            {json.dumps(json_safe(self.results), indent=2, default=str)[:8000] or '(empty)'}

            EXECUTION LOGS
            {self._tail_logs()}

            FAILED STEPS
            {json.dumps(self.failures, indent=2) if self.failures else '(none)'}

            CHARTS PRODUCED
            {json.dumps([c.title for c in self.plots])}

            USER GOAL
            {self.user_goal or '(none supplied)'}

            Write the report.
            """
        ).strip()
        self.report = self.client.complete(  # type: ignore[union-attr]
            REPORTER_SYSTEM, [{"role": "user", "content": prompt}], temperature=0.3
        )
        yield AgentEvent("report", "report", self.report, {"report": self.report})

    # -- public API -------------------------------------------------------- #
    def run(self) -> Iterator[AgentEvent]:
        """Run the full pipeline, yielding events for live rendering."""
        started = time.time()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.client = ClaudeClient(self.config)
        except Exception as exc:
            yield AgentEvent("error", "init", str(exc))
            return

        try:
            yield from self._stage_profile()
            yield from self._stage_framing()
            yield from self._stage_analyze()
            yield from self._stage_eda()
            yield from self._stage_charts()
            yield from self._stage_prepare()
            yield from self._stage_model()
            yield from self._stage_report()
        except Exception as exc:  # pragma: no cover - top-level safety net
            yield AgentEvent("error", "pipeline", f"{type(exc).__name__}: {exc}")
            LOGGER.exception("Pipeline aborted")

        elapsed = time.time() - started
        yield AgentEvent(
            "done",
            "done",
            f"Run finished in {elapsed:.1f}s across {len(self.transcript)} executed cells.",
            {
                "elapsed": elapsed,
                "cells": len(self.transcript),
                "plots": self.plots,
                "failures": len(self.failures),
                "input_tokens": getattr(self.client, "input_tokens", 0),
                "output_tokens": getattr(self.client, "output_tokens", 0),
            },
        )

    def bundle(self) -> Dict[str, Any]:
        """Everything the notebook builder and the UI need after a run."""
        return {
            "run_id": self.run_id,
            "spec": json_safe(self.spec),
            "results": json_safe(self.results),
            "report": self.report,
            "chart_plan": json_safe(self.chart_plan),
            "framing": json_safe(self.framing),
            "transcript": self.transcript,
            "plots": self.plots,
            "failures": self.failures,
            "model": self.config.model,
            "random_state": self.config.random_state,
            "user_goal": self.user_goal,
        }


__all__ = [
    "AgentConfig",
    "AgentEvent",
    "AutoDSAgent",
    "ChartArtifact",
    "ClaudeClient",
    "DEFAULT_MODEL",
    "ExecResult",
    "PythonREPL",
    "SUPPORTED_MODELS",
    "TranscriptCell",
    "extract_code",
    "extract_json",
]
