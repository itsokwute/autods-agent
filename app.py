"""
AutoDS-Agent :: Streamlit Frontend
==================================

    streamlit run app.py

Four panes: ingest anything, watch the agent work in real time, read the
report, export an executable Colab notebook.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from agent import DEFAULT_MODEL, SUPPORTED_MODELS, AgentConfig, AutoDSAgent
from ingestion import IngestionResult, ingest_many, json_safe
from notebook_builder import build_notebook, notebook_to_bytes, suggested_filename

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="AutoDS-Agent",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
      .autods-title {font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.1rem;}
      .autods-sub {color: #6b7280; font-size: 0.95rem; margin-bottom: 1.4rem;}
      .stream-line {font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem;}
      div[data-testid="stMetricValue"] {font-size: 1.45rem;}
      .stTabs [data-baseweb="tab"] {padding-top: 0.6rem; padding-bottom: 0.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

ACCEPTED_TYPES = ["csv", "tsv", "txt", "xlsx", "xls", "xlsm", "json", "jsonl", "yaml", "yml", "pdf", "docx", "zip", "parquet"]

STAGE_LABEL = {
    "profile": "Profiling",
    "framing": "Business Framing",
    "analyze": "Problem Analysis",
    "eda": "EDA",
    "charts": "Charts",
    "prepare": "Preprocessing",
    "model": "Modelling",
    "report": "Report",
    "done": "Complete",
    "init": "Init",
    "pipeline": "Pipeline",
}
KIND_ICON = {
    "status": "•",
    "code": "▸",
    "stdout": " ",
    "error": "✕",
    "repair": "↻",
    "plot": "▣",
    "spec": "◆",
    "framing": "💡",
    "report": "✎",
    "done": "✔",
}


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state() -> None:
    defaults: Dict[str, Any] = {
        "ingestion": None,
        "frames": {},
        "texts": {},
        "bundle": None,
        "events": [],
        "running": False,
        "run_seconds": 0.0,
        "notebook_bytes": None,
        "notebook_name": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _secret(name: str, default: str = "") -> str:
    """Env var first, then .streamlit/secrets.toml (used by Streamlit Cloud)."""
    value = os.getenv(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:      # no secrets.toml present -- normal for local runs
        return default


SECRETS_PATH = Path(__file__).parent / ".streamlit" / "secrets.toml"

# Hosted deployments behave differently: no writing secrets to an ephemeral disk,
# and a run cap when the operator supplies the API key.
#   HOSTED        explicit switch -- set it as a secret on any host
#   SPACE_ID      Hugging Face Spaces
#   /mount/src    Streamlit Community Cloud mounts the repo here
IS_HOSTED = bool(
    _secret("HOSTED")
    or os.getenv("SPACE_ID")
    or Path("/mount/src").exists()
)


def clean_key(raw: str) -> str:
    """Strip what copy-paste adds: whitespace, quotes, newlines, zero-width chars.

    Pasting from a console, an email, or a secrets box routinely carries a trailing
    newline or a pair of quotes. The API rejects those with a bare 401 that names
    no cause, so remove them before the key is ever used.
    """
    if not raw:
        return ""
    cleaned = str(raw).strip()
    for ch in ("\u200b", "\u200c", "\u200d", "\ufeff", "\xa0", "\n", "\r", "\t", " "):
        cleaned = cleaned.replace(ch, "")
    if len(cleaned) >= 2 and cleaned[0] in "\"'\u201c\u2018" and cleaned[-1] in "\"'\u201d\u2019":
        cleaned = cleaned[1:-1].strip()
    return cleaned


def inspect_key(raw: str) -> list:
    """(passed, message) checks on the key's shape. Never reveals the key."""
    cleaned = clean_key(raw)
    checks = []
    if raw and raw != cleaned:
        removed = len(raw) - len(cleaned)
        checks.append((True, f"Removed {removed} stray character(s) from your paste (spaces or quotes)."))
    checks.append((bool(cleaned), "Key is present." if cleaned else "No key entered."))
    if not cleaned:
        return checks
    checks.append((cleaned.startswith("sk-ant-"),
                   "Starts with sk-ant-." if cleaned.startswith("sk-ant-")
                   else f"Does NOT start with sk-ant- (starts with '{cleaned[:7]}'). This is not an Anthropic API key."))
    long_enough = len(cleaned) >= 90
    checks.append((long_enough,
                   f"Length {len(cleaned)} characters — looks complete."
                   if long_enough else
                   f"Length {len(cleaned)} — too short. A full key is about 100+ characters, so the copy was cut off."))
    checks.append((cleaned.isascii(), "No unusual characters." if cleaned.isascii()
                   else "Contains non-standard characters — retype it rather than pasting."))
    return checks


def save_key(key: str) -> tuple[bool, str]:
    """Write the key to .streamlit/secrets.toml so it is never typed again."""
    try:
        SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = SECRETS_PATH.read_text() if SECRETS_PATH.exists() else ""
        lines = [ln for ln in existing.splitlines() if not ln.strip().startswith("ANTHROPIC_API_KEY")]
        lines.insert(0, f'ANTHROPIC_API_KEY = "{clean_key(key)}"')
        SECRETS_PATH.write_text("\n".join(lines).strip() + "\n")
        return True, f"Saved. It will load automatically from {SECRETS_PATH.name} from now on."
    except Exception as exc:
        return False, f"Could not save the key: {exc}"


def forget_key() -> tuple[bool, str]:
    try:
        if SECRETS_PATH.exists():
            kept = [
                ln for ln in SECRETS_PATH.read_text().splitlines()
                if not ln.strip().startswith("ANTHROPIC_API_KEY")
            ]
            SECRETS_PATH.write_text("\n".join(kept).strip() + "\n" if any(kept) else "")
        return True, "Saved key removed from this computer."
    except Exception as exc:
        return False, f"Could not remove the key: {exc}"


# Plain-English translation of the errors this app can actually produce.
# (pattern to look for, what happened, what to do about it)
ERROR_HELP: List[tuple] = [
    ("credit balance",
     "Your Anthropic account is out of credit.",
     "Go to console.anthropic.com → Billing → add credit. $5 is plenty. Then run again."),
    ("authentication_error",
     "The API key was rejected.",
     "Check for a stray space at the start or end. If it still fails, create a new key at "
     "console.anthropic.com → Settings → API keys, paste it in the sidebar, and click Save key."),
    ("invalid x-api-key",
     "The API key was rejected.",
     "Create a new key at console.anthropic.com → Settings → API keys, paste it in the sidebar, "
     "and click Save key."),
    ("permission_error",
     "This key doesn't have access to the selected model.",
     "Pick a different model in the sidebar, or check your account's model access."),
    ("not_found_error",
     "The selected model isn't available to your account.",
     "Choose another model from the Planner model dropdown in the sidebar."),
    ("rate_limit",
     "Too many requests too quickly.",
     "Wait about a minute and run again. The app already retries automatically."),
    ("overloaded",
     "Anthropic's servers are busy right now.",
     "Wait a minute and run again. Nothing is wrong on your end."),
    ("temperature",
     "This model rejects a setting the app sent.",
     "The app corrects this automatically. If you still see it, you're on an older copy of "
     "agent.py — re-download it."),
    ("connection", "Couldn't reach the internet.", "Check your connection, then run again."),
    ("timed out", "The request took too long.", "Run again. If it repeats, pick a smaller dataset."),
    ("max_tokens",
     "The response hit the length limit.",
     "Lower 'Max output tokens per call' in the sidebar, or use a smaller dataset."),
]


def explain_error(message: str) -> tuple[str, str]:
    """Return (what happened, what to do) for a raw error string."""
    lowered = str(message).lower()
    for pattern, what, fix in ERROR_HELP:
        if pattern in lowered:
            return what, fix
    return "Something went wrong.", "The full technical message is below. Send it to Claude if you're stuck."


def test_key(key: str, model: str) -> tuple[bool, str]:
    """One tiny API call that proves the key, the credit, and the model all work."""
    key = clean_key(key)
    if not key:
        return False, "No key entered."
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        client.messages.create(
            model=model, max_tokens=1, messages=[{"role": "user", "content": "hi"}]
        )
        return True, "Key works. Credit and model access confirmed."
    except Exception as exc:
        what, fix = explain_error(str(exc))
        return False, f"{what}\n\n{fix}"


def run_diagnostics() -> List[tuple]:
    """(passed, label, fix) for every precondition the app needs."""
    import importlib.util

    checks: List[tuple] = []
    for module, package in [
        ("streamlit", "streamlit"), ("anthropic", "anthropic"), ("pandas", "pandas"),
        ("sklearn", "scikit-learn"), ("lightgbm", "lightgbm"), ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"), ("pdfplumber", "pdfplumber"), ("docx", "python-docx"),
        ("yaml", "PyYAML"),
    ]:
        ok = importlib.util.find_spec(module) is not None
        checks.append((ok, package, "" if ok else f"Not installed. Close the app and run: pip install {package}"))
    return checks


def _password_gate() -> None:
    """Optional gate. Set APP_PASSWORD when the app is reachable from the internet.

    This stops casual visitors from spending your API budget and running code in
    your container. It is not a substitute for the container isolation described
    in README.md -- anyone who gets past it can still execute generated Python.
    """
    expected = _secret("APP_PASSWORD")
    if not expected:
        return
    if st.session_state.get("authenticated"):
        return
    st.markdown("### 🔒 AutoDS-Agent")
    st.caption("This deployment is password protected.")
    entered = st.text_input("Password", type="password", key="pw_input")
    if st.button("Enter"):
        if entered == expected:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


_password_gate()

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    env_key = _secret("ANTHROPIC_API_KEY")
    api_key = st.text_input(
        "Anthropic API key",
        value=env_key,
        type="password",
        help="Loads automatically once saved. Click 'Save key' below to store it on this computer.",
    )
    api_key = clean_key(api_key)

    if env_key and api_key == env_key:
        st.caption("✓ Key loaded automatically — you don't need to type it.")

    model = st.selectbox("Planner model", SUPPORTED_MODELS, index=SUPPORTED_MODELS.index(DEFAULT_MODEL))

    using_server_key = bool(env_key) and api_key == env_key

    if IS_HOSTED:
        # Saving to disk is meaningless on an ephemeral container -- offer only the test.
        if st.button("Test key", use_container_width=True, disabled=not api_key):
            with st.spinner("Checking..."):
                ok, message = test_key(api_key, model)
            (st.success if ok else st.error)(message)
    else:
        key_cols = st.columns([1, 1, 1])
        if key_cols[0].button("Save key", use_container_width=True, disabled=not api_key):
            ok, message = save_key(api_key)
            (st.success if ok else st.error)(message)
        if key_cols[1].button("Test key", use_container_width=True, disabled=not api_key):
            with st.spinner("Checking..."):
                ok, message = test_key(api_key, model)
            (st.success if ok else st.error)(message)
        if key_cols[2].button("Forget", use_container_width=True):
            ok, message = forget_key()
            (st.success if ok else st.error)(message)

    with st.expander("🩺 Diagnostics", expanded=False):
        st.caption("Run this first if anything misbehaves.")
        checks = run_diagnostics()
        failed = [c for c in checks if not c[0]]
        if not failed:
            st.success(f"All {len(checks)} libraries installed.")
        else:
            for _, label, fix in failed:
                st.error(f"**{label}** — {fix}")
        for passed, message in inspect_key(st.session_state.get("api_key_raw", api_key)):
            (st.success if passed else st.error)(message)
        if SECRETS_PATH.exists():
            st.caption(f"Key file: {SECRETS_PATH}")
        st.caption(f"Working folder: {Path(__file__).parent}")

    with st.expander("Agent behaviour", expanded=False):
        max_repairs = st.slider("Self-correction attempts per stage", 0, 6, 3)
        timeout = st.slider("Per-cell timeout (seconds)", 60, 1800, 600, step=30)
        sample_rows = st.slider("Sample rows shown to the planner", 3, 20, 5)
        random_state = st.number_input("Random state", value=42, step=1)
        enable_lgb = st.checkbox("Train LightGBM alongside scikit-learn", value=True)
        n_charts = st.slider("Number of charts to generate", 3, 8, 5)
        max_tokens = st.select_slider("Max output tokens per call", [2000, 4000, 8000, 16000], value=8000)

    with st.expander("Analysis scope", expanded=True):
        if "queued_goal" in st.session_state:
            st.session_state["user_goal"] = st.session_state.pop("queued_goal")
        user_goal = st.text_area(
            "Business goal (optional)",
            placeholder="e.g. Predict which wafers will fail final parametric test so we can pull them early.",
            height=90,
            key="user_goal",
        )
        target_override = st.text_input(
            "Force target column (optional)",
            placeholder="leave blank to let the agent choose",
        )

    st.divider()
    if st.session_state.bundle:
        b = st.session_state.bundle
        st.caption(
            f"Last run `{b['run_id']}` · {len(b.get('transcript', []))} cells · "
            f"{len(b.get('failures', []))} repaired errors"
        )
    st.caption("Generated code executes locally. Review it before trusting any result.")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown('<div class="autods-title">AutoDS-Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="autods-sub">Ingest anything → scope the problem → execute, self-correct, model → '
    "export a runnable Colab notebook.</div>",
    unsafe_allow_html=True,
)

try:
    MAX_RUNS = int(_secret("MAX_RUNS_PER_SESSION", "3"))
except ValueError:
    MAX_RUNS = 3

if IS_HOSTED:
    if not env_key:
        st.info(
            "**Live demo.** Paste your own Anthropic API key in the sidebar to run an analysis — "
            "get one free at [console.anthropic.com](https://console.anthropic.com). "
            "Your key is used for this session only and is never stored. "
            "Uploaded files are processed in memory and wiped when the app restarts. "
            "On free hosting, keep demo files under about 50 MB."
        )
    elif using_server_key:
        remaining = max(MAX_RUNS - st.session_state.get("runs_used", 0), 0)
        st.info(
            f"**Live demo.** {remaining} of {MAX_RUNS} free runs left this session. "
            "Paste your own Anthropic API key in the sidebar for unlimited runs. "
            "Uploaded files are wiped when the app restarts. "
            "On free hosting, keep demo files under about 50 MB."
        )


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
with st.container(border=True):
    st.markdown("#### 1 · Load data")
    left, right = st.columns([3, 2])
    with left:
        uploads = st.file_uploader(
            "Files",
            type=ACCEPTED_TYPES,
            accept_multiple_files=True,
            help="CSV · XLSX · JSON · YAML · PDF · DOCX · ZIP · Parquet. ZIPs are unpacked recursively.",
        )
    with right:
        url_text = st.text_area(
            "URLs (one per line)",
            placeholder="https://example.com/data.csv",
            height=100,
        )

    if st.button("Ingest", type="primary", use_container_width=False):
        payloads = [(f.name, f.getvalue()) for f in (uploads or [])]
        urls = [u for u in (url_text or "").splitlines() if u.strip()]
        if not payloads and not urls:
            st.warning("Add at least one file or URL.")
        else:
            with st.spinner("Parsing inputs..."):
                result: IngestionResult = ingest_many(payloads, urls)
            st.session_state.ingestion = result
            st.session_state.frames = result.frames
            st.session_state.texts = {t.name: t.text for t in result.texts}
            st.session_state.bundle = None
            st.session_state.events = []
            st.session_state.notebook_bytes = None
            if result.is_empty:
                st.error("No tabular data could be recovered. Check the warnings below.")
            else:
                st.success(f"Recovered {len(result.frames)} table(s) and {len(result.texts)} text artifact(s).")

result: IngestionResult | None = st.session_state.ingestion
frames: Dict[str, pd.DataFrame] = st.session_state.frames

if result and result.warnings:
    with st.expander(f"Ingestion warnings ({len(result.warnings)})"):
        for warning in result.warnings:
            st.caption(f"• {warning}")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_data, tab_business, tab_run, tab_plots, tab_report, tab_export = st.tabs(
    ["📊 Data", "💡 Business", "🤖 Agent run", "📈 Plots", "📝 Report", "📓 Export"]
)


FEASIBILITY_BADGE = {
    "answerable": ("✅", "Answerable with this data"),
    "partial": ("⚠️", "Partially — needs an assumption"),
    "not_answerable": ("❌", "Not answerable — needs more data"),
}


def render_framing_item(item: dict, kind: str, index: int, table: str) -> None:
    """One business question or problem, with feasibility and a run button."""
    headline = item.get("question") if kind == "question" else item.get("problem")
    feas = str(item.get("feasibility", "")).lower()
    icon, label = FEASIBILITY_BADGE.get(feas, ("•", ""))

    st.markdown(f"**{index}. {headline}**")
    st.caption(f"{icon} {label}")

    detail_cols = st.columns([3, 2])
    with detail_cols[0]:
        if kind == "question":
            if item.get("why_it_matters"):
                st.markdown(f"*Why it matters:* {item['why_it_matters']}")
            if item.get("analysis_needed"):
                st.markdown(f"*Analysis needed:* {item['analysis_needed']}")
        else:
            if item.get("who_owns_it"):
                st.markdown(f"*Owned by:* {item['who_owns_it']}")
            if item.get("decision_it_changes"):
                st.markdown(f"*Changes this decision:* {item['decision_it_changes']}")
            if item.get("how_to_measure_improvement"):
                st.markdown(f"*Measured by:* {item['how_to_measure_improvement']}")
        if item.get("feasibility_note"):
            st.markdown(f"*Note:* {item['feasibility_note']}")
    with detail_cols[1]:
        columns_used = item.get("columns_used") or []
        if columns_used:
            st.caption("Columns: " + ", ".join(f"`{c}`" for c in columns_used[:8]))

    if feas != "not_answerable" and headline:
        if st.button("Analyze this", key=f"goal_{table}_{kind}_{index}", use_container_width=False):
            st.session_state["queued_goal"] = str(headline)
            st.session_state["goal_notice"] = str(headline)
            st.rerun()
    st.divider()


# ---------------------------------- Data ----------------------------------- #
with tab_data:
    if not frames:
        st.info("Load data to see the preview and summary metrics.")
    else:
        total_rows = sum(len(f) for f in frames.values())
        total_cols = sum(f.shape[1] for f in frames.values())
        missing = (
            sum(f.isna().sum().sum() for f in frames.values())
            / max(sum(f.size for f in frames.values()), 1)
            * 100
        )
        memory = sum(f.memory_usage(deep=True).sum() for f in frames.values()) / 1e6

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Tables", len(frames))
        c2.metric("Total rows", f"{total_rows:,}")
        c3.metric("Total columns", total_cols)
        c4.metric("Missing", f"{missing:.1f}%")
        c5.metric("Memory", f"{memory:.1f} MB")

        st.dataframe(result.summary(), use_container_width=True, hide_index=True)

        selected = st.selectbox("Preview table", list(frames.keys()), index=0)
        frame = frames[selected]
        st.dataframe(frame.head(200), use_container_width=True, height=340)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Schema**")
            schema = pd.DataFrame(
                {
                    "column": frame.columns,
                    "dtype": [str(d) for d in frame.dtypes],
                    "missing_%": (frame.isna().mean() * 100).round(2).values,
                    "unique": [int(frame[c].nunique(dropna=True)) for c in frame.columns],
                }
            )
            st.dataframe(schema, use_container_width=True, hide_index=True, height=280)
        with col_b:
            st.markdown("**Numeric summary**")
            numeric = frame.select_dtypes("number")
            if numeric.empty:
                st.caption("No numeric columns in this table.")
            else:
                st.dataframe(numeric.describe().T.round(3), use_container_width=True, height=280)

        if st.session_state.texts:
            with st.expander(f"Document text extracted ({len(st.session_state.texts)})"):
                for name, body in st.session_state.texts.items():
                    st.markdown(f"**{name}**")
                    st.text(body[:2500] + ("\n... [truncated]" if len(body) > 2500 else ""))



# -------------------------------- Business --------------------------------- #
with tab_business:
    bundle = st.session_state.bundle
    framing = (bundle or {}).get("framing") or {}
    if "goal_notice" in st.session_state:
        st.success(
            f"Goal set: *{st.session_state.pop('goal_notice')}*  \n"
            "Open **🤖 Agent run** and click **▶ Run agent**."
        )
    if not framing and not bundle:
        st.info(
            "After a run, this tab lists the business questions and business problems "
            "each dataset can support — with an honest note when it supports fewer than five."
        )
    elif not framing:
        st.warning(
            "The run finished but produced no business framing.\n\n"
            "Open **🤖 Agent run** and look for a **Business Framing** stage near the top. "
            "If it isn't there, `agent.py` wasn't replaced — re-download it and drop it into "
            "the AutoDS-Agent folder, checking the name isn't `agent (1).py`. "
            "If it is there but shows a red error, the message names the cause."
        )
    else:
        table_names = list(framing.keys())
        chosen = table_names[0] if len(table_names) == 1 else st.selectbox("Dataset", table_names)
        block = framing.get(chosen, {})

        if block.get("dataset_summary"):
            st.markdown(f"**{block['dataset_summary']}**")
        if block.get("domain_guess"):
            st.caption(f"Likely domain: {block['domain_guess']}")

        questions = block.get("business_questions") or []
        problems = block.get("business_problems") or []
        target = 5

        st.markdown("### Business questions")
        if block.get("questions_shortfall"):
            st.warning(f"Only {len(questions)} of {target}. {block['questions_shortfall']}")
        elif questions:
            st.caption(f"{len(questions)} questions this dataset can answer.")
        for i, item in enumerate(questions, start=1):
            render_framing_item(item, "question", i, chosen)
        if not questions:
            st.error("No business questions could be supported by this dataset.")

        st.markdown("### Business problems")
        if block.get("problems_shortfall"):
            st.warning(f"Only {len(problems)} of {target}. {block['problems_shortfall']}")
        elif problems:
            st.caption(f"{len(problems)} problems this dataset could help solve.")
        for i, item in enumerate(problems, start=1):
            render_framing_item(item, "problem", i, chosen)
        if not problems:
            st.error("No business problems could be supported by this dataset.")


# ------------------------------- Agent run --------------------------------- #
def render_event(event) -> None:
    stage = STAGE_LABEL.get(event.stage, event.stage)
    icon = KIND_ICON.get(event.kind, "•")
    if event.kind == "code":
        with st.expander(f"{icon} {stage} — generated code (attempt {(event.data or {}).get('attempt', 1)})"):
            st.code(event.content, language="python")
    elif event.kind == "stdout":
        with st.expander(f"  {stage} — output", expanded=False):
            st.code(event.content or "(no output)", language="text")
    elif event.kind == "error":
        if event.stage in ("init", "pipeline"):
            what, fix = explain_error(event.content)
            st.error(f"**{what}**\n\n{fix}")
            with st.expander("Technical details"):
                st.code(event.content, language="text")
        else:
            st.error(f"{stage}: {event.content}")
    elif event.kind == "repair":
        st.warning(f"{icon} {event.content}")
    elif event.kind == "framing":
        data = event.data or {}
        framing = data.get("framing", {})
        nq = len(framing.get("business_questions") or [])
        npb = len(framing.get("business_problems") or [])
        with st.expander(f"💡 {data.get('table', '')} — {nq} question(s), {npb} problem(s)", expanded=False):
            st.caption(framing.get("dataset_summary", ""))
            for item in (framing.get("business_questions") or [])[:3]:
                st.markdown(f"- {item.get('question', '')}")
            if nq > 3:
                st.caption(f"...and {nq - 3} more — see the 💡 Business tab.")
    elif event.kind == "spec":
        with st.expander("◆ Analysis specification", expanded=True):
            st.json(json.loads(event.content))
    elif event.kind == "plot":
        st.caption(f"{icon} chart: {event.content}")
    elif event.kind == "report":
        st.success("✎ Report synthesised — see the Report tab.")
    elif event.kind == "done":
        st.success(f"{icon} {event.content}")
    else:
        st.markdown(f'<div class="stream-line">{icon} <b>{stage}</b> — {event.content}</div>', unsafe_allow_html=True)


with tab_run:
    ready = bool(frames)
    col_run, col_note = st.columns([1, 4])
    with col_run:
        launch = st.button("▶ Run agent", type="primary", disabled=not ready or st.session_state.running)
    with col_note:
        if not ready:
            st.caption("Load data first.")
        elif not api_key:
            st.caption("An Anthropic API key is required in the sidebar.")

    # One gate, evaluated in order: demo cap -> key present -> key actually works.
    # The key is verified with a 1-token call BEFORE the run, so an invalid key
    # fails in two seconds with a clear message instead of after every stage.
    cleared_to_run = False
    if launch:
        if IS_HOSTED and using_server_key and st.session_state.get("runs_used", 0) >= MAX_RUNS:
            st.error(
                f"Demo limit reached — {MAX_RUNS} runs per session on the shared key.\n\n"
                "Paste your own Anthropic API key in the sidebar to keep going. "
                "Get one at console.anthropic.com."
            )
        elif not api_key:
            st.error("Add your Anthropic API key in the sidebar.")
        else:
            with st.spinner("Checking your API key..."):
                key_ok, key_message = test_key(api_key, model)
            if key_ok:
                cleared_to_run = True
            else:
                st.error(f"**The run did not start — no credit was used.**\n\n{key_message}")
                st.info(
                    "Open **🩺 Diagnostics** in the sidebar. It inspects your key and names "
                    "exactly what is wrong with it — wrong prefix, cut-off paste, or stray characters."
                )

    if cleared_to_run:
        if True:
            config = AgentConfig(
                api_key=api_key,
                model=model,
                max_tokens=int(max_tokens),
                max_repair_attempts=int(max_repairs),
                sample_rows=int(sample_rows),
                execution_timeout=int(timeout),
                random_state=int(random_state),
                enable_lightgbm=bool(enable_lgb),
                n_charts=int(n_charts),
            )
            agent = AutoDSAgent(
                config=config,
                frames=frames,
                texts=st.session_state.texts,
                user_goal=user_goal,
                target_override=(target_override.strip() or None),
            )

            st.session_state.running = True
            st.session_state.events = []
            started = time.time()
            progress = st.progress(0.0, text="Starting...")
            stream = st.container()
            stage_order = ["profile", "framing", "analyze", "eda", "charts", "prepare", "model", "report", "done"]

            try:
                for event in agent.run():
                    st.session_state.events.append(event)
                    if event.stage in stage_order:
                        idx = stage_order.index(event.stage)
                        progress.progress(
                            (idx + 1) / len(stage_order),
                            text=f"{STAGE_LABEL.get(event.stage, event.stage)}...",
                        )
                    with stream:
                        render_event(event)
            finally:
                st.session_state.running = False
                st.session_state.run_seconds = time.time() - started
                st.session_state.bundle = agent.bundle()
                st.session_state.notebook_bytes = None
                st.session_state.runs_used = st.session_state.get("runs_used", 0) + 1
                progress.progress(1.0, text="Finished.")

            # Streamlit renders the script top to bottom, so the Business, Plots and
            # Report tabs were already drawn (empty) before this run finished. Switching
            # tabs is client-side and does not re-run the script, so without this the
            # results never appear. The log is replayed from session_state.events.
            st.rerun()

    elif st.session_state.events:
        st.caption(f"Last run · {st.session_state.run_seconds:.0f}s · {len(st.session_state.events)} events")
        for event in st.session_state.events:
            render_event(event)


# --------------------------------- Plots ----------------------------------- #
with tab_plots:
    bundle = st.session_state.bundle
    transcript = (bundle or {}).get("transcript", [])
    pairs = [
        (cell, art)
        for cell in transcript
        for art in getattr(cell, "plots", [])
        if Path(art.path).exists()
    ]
    if not pairs and not bundle:
        st.info("Charts generated during the run appear here, one per row.")
    elif not pairs:
        st.warning("The run finished but produced no charts. Check 🤖 Agent run for errors in the Charts stage.")
    else:
        st.caption(f"{len(pairs)} chart(s). Each is full width with the code that produced it.")
        plan = (bundle or {}).get("chart_plan") or {}
        for number, (cell, art) in enumerate(pairs, start=1):
            st.markdown(f"#### {number}. {art.title}")
            st.image(art.path, use_container_width=True)
            if art.caption and art.caption.strip() != art.title.strip():
                st.caption(art.caption)
            with st.expander("Code that produced this chart"):
                st.code(cell.code, language="python")
            if cell.stdout:
                with st.expander("Printed output"):
                    st.code(cell.stdout, language="text")
            st.divider()
        if plan.get("skipped"):
            with st.expander(f"Charts considered and rejected ({len(plan['skipped'])})"):
                for item in plan["skipped"]:
                    st.caption(f"• {item}")


# --------------------------------- Report ---------------------------------- #
with tab_report:
    bundle = st.session_state.bundle
    if not bundle:
        st.info("The conclusion and future-work report appears here after a run.")
    else:
        spec = bundle.get("spec") or {}
        results = bundle.get("results") or {}
        if spec:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Paradigm", str(spec.get("ml_paradigm", "—")).replace("_", " "))
            m2.metric("Target", str(spec.get("target_variable") or "—"))
            m3.metric("Metric", str(spec.get("evaluation_metric") or "—"))
            m4.metric("Best model", str(results.get("best_model", "—")))

        if bundle.get("report"):
            st.markdown(bundle["report"])
        else:
            st.warning("No report was produced — check the agent run tab for failures.")

        if results:
            with st.expander("Raw metrics (RESULTS)"):
                st.json(json_safe(results))
        if spec:
            with st.expander("Analysis specification"):
                st.json(json_safe(spec))
        if bundle.get("failures"):
            with st.expander(f"Self-correction log ({len(bundle['failures'])})"):
                st.dataframe(pd.DataFrame(bundle["failures"]), use_container_width=True, hide_index=True)


# --------------------------------- Export ---------------------------------- #
with tab_export:
    bundle = st.session_state.bundle
    if not bundle:
        st.info("Run the agent to enable notebook export.")
    else:
        st.markdown(
            "Exports a **self-contained Colab notebook**: every executed cell in order, with the "
            "ingested tables embedded as gzip+base64 CSV so it reproduces without the app."
        )
        include_pip = st.checkbox("Include the `%pip install` bootstrap cell", value=True)

        if st.button("Build notebook", type="primary"):
            with st.spinner("Assembling notebook..."):
                notebook = build_notebook(frames, bundle, include_pip_cell=include_pip)
                st.session_state.notebook_bytes = notebook_to_bytes(notebook)
                st.session_state.notebook_name = suggested_filename(bundle)
            st.success(f"{len(notebook['cells'])} cells · {len(st.session_state.notebook_bytes) / 1e6:.2f} MB")

        if st.session_state.notebook_bytes:
            st.download_button(
                "⬇ Download .ipynb",
                data=st.session_state.notebook_bytes,
                file_name=st.session_state.notebook_name,
                mime="application/x-ipynb+json",
                type="primary",
            )
            st.caption("Upload to colab.research.google.com and choose Runtime → Run all.")
