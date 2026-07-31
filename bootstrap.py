#!/usr/bin/env python3
"""
AutoDS-Agent :: Bootstrap
=========================

Turns a pile of downloaded files into a working project folder, then tells you
exactly what is still missing.

    python bootstrap.py              # find, place, and verify everything
    python bootstrap.py --check      # verify only, change nothing
    python bootstrap.py --from ~/Downloads/autods

Why this exists
---------------
Browsers mangle the files that matter most. `.gitignore` arrives as
`gitignore.txt`, `.env.example` loses its leading dot, and `config.toml` and
`secrets.toml.example` both land flat in Downloads instead of inside a
`.streamlit/` folder. On Windows, Explorer actively fights you when you try to
create a filename that starts with a dot.

This script handles all of that: it matches each downloaded file to where it
belongs (by name, and by content when names collide), creates the folder
structure, and runs a full readiness check.

Standard library only -- run it before installing anything.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# What a complete project looks like
# --------------------------------------------------------------------------- #
# target path -> (required?, a phrase that must appear in the real file)
EXPECTED: Dict[str, Tuple[bool, str]] = {
    "app.py":                        (True,  "streamlit"),
    "agent.py":                      (True,  "AutoDSAgent"),
    "ingestion.py":                  (True,  "IngestionResult"),
    "notebook_builder.py":           (True,  "build_notebook"),
    "requirements.txt":              (True,  "streamlit"),
    "README.md":                     (False, "AutoDS-Agent"),
    "SETUP.md":                      (False, "Setup"),
    "BUILD.md":                      (False, "Zero to a Working Tool"),
    "LICENSE":                       (False, "MIT"),
    "Dockerfile":                    (False, "streamlit"),
    "packages.txt":                  (False, "libgomp"),
    ".gitignore":                    (True,  ".env"),
    ".dockerignore":                 (False, "__pycache__"),
    ".env.example":                  (False, "ANTHROPIC_API_KEY"),
    ".streamlit/config.toml":        (False, "[server]"),
    ".streamlit/secrets.toml.example": (False, "ANTHROPIC_API_KEY"),
}

CORE = ["app.py", "agent.py", "ingestion.py", "notebook_builder.py"]

# Downloaded-name variants -> target path. Matching is case-insensitive and
# ignores browser suffixes like " (1)" and a trailing ".txt".
ALIASES: Dict[str, str] = {
    "app": "app.py",
    "agent": "agent.py",
    "ingestion": "ingestion.py",
    "notebook_builder": "notebook_builder.py",
    "notebookbuilder": "notebook_builder.py",
    "notebook builder": "notebook_builder.py",
    "requirements": "requirements.txt",
    "readme": "README.md",
    "setup": "SETUP.md",
    "build": "BUILD.md",
    "license": "LICENSE",
    "dockerfile": "Dockerfile",
    "packages": "packages.txt",
    "gitignore": ".gitignore",
    "dockerignore": ".dockerignore",
    "env": ".env.example",
    "env.example": ".env.example",
    "config": ".streamlit/config.toml",
    "secrets": ".streamlit/secrets.toml.example",
    "secrets.toml": ".streamlit/secrets.toml.example",
    "bootstrap": None,   # this script; ignore it
}

RUNTIME_IMPORTS = [
    ("streamlit", "streamlit"),
    ("anthropic", "anthropic"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("lightgbm", "lightgbm"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("pdfplumber", "pdfplumber"),
    ("docx", "python-docx"),
    ("yaml", "PyYAML"),
    ("requests", "requests"),
]

OK, WARN, BAD, INFO = "  [ok]  ", "  [--]  ", "  [XX]  ", "  [ ]   "


# --------------------------------------------------------------------------- #
# Console helpers (no colour: Windows terminals mangle ANSI often enough)
# --------------------------------------------------------------------------- #
def head(title: str) -> None:
    print(f"\n{title}\n" + "-" * max(40, len(title)))


def normalise(filename: str) -> str:
    """'Requirements (1).txt' -> 'requirements'; 'gitignore.txt' -> 'gitignore'."""
    stem = filename.strip()
    stem = re.sub(r"\s*\(\d+\)", "", stem)          # browser duplicate suffix
    stem = re.sub(r"\.txt$", "", stem, flags=re.I)  # browsers append .txt a lot
    stem = re.sub(r"\.(md|py|toml|example)$", "", stem, flags=re.I)
    return stem.strip().lstrip(".").lower()


def classify(path: Path) -> Optional[str]:
    """Work out where a downloaded file belongs. Content breaks name ties."""
    name = path.name

    # Exact matches win outright.
    if name in EXPECTED:
        return name
    for target in EXPECTED:
        if Path(target).name == name:
            return target

    key = normalise(name)
    if key in ALIASES:
        target = ALIASES[key]
        if target is None:
            return None
        # 'config'/'secrets' both look like toml -- confirm with content.
        if target.startswith(".streamlit"):
            try:
                body = path.read_text(errors="ignore")[:800]
            except Exception:
                return target
            if "[server]" in body or "[theme]" in body:
                return ".streamlit/config.toml"
            if "ANTHROPIC_API_KEY" in body:
                return ".streamlit/secrets.toml.example"
        return target

    # Last resort: sniff the content of stray .py files.
    if path.suffix == ".py":
        try:
            body = path.read_text(errors="ignore")[:4000]
        except Exception:
            return None
        for target, (_, marker) in EXPECTED.items():
            if target.endswith(".py") and marker in body:
                return target
    return None


def candidate_sources(explicit: Optional[str]) -> List[Path]:
    if explicit:
        return [Path(explicit).expanduser()]
    home = Path.home()
    return [
        Path.cwd(),
        home / "Downloads",
        home / "Desktop",
        home / "Downloads" / "AutoDS-Agent",
        home / "Desktop" / "AutoDS-Agent",
    ]


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def collect(project: Path, sources: List[Path], dry_run: bool) -> Dict[str, Path]:
    """Copy every recognised file from the source folders into the project."""
    placed: Dict[str, Path] = {}

    for source in sources:
        if not source.is_dir():
            continue
        try:
            entries = sorted(source.iterdir())
        except PermissionError:
            continue
        for entry in entries:
            if not entry.is_file() or entry.name == "bootstrap.py":
                continue
            target_rel = classify(entry)
            if target_rel is None or target_rel in placed:
                continue

            target = project / target_rel
            if target.exists() and target.resolve() == entry.resolve():
                placed[target_rel] = target       # already in the right place
                continue

            if dry_run:
                print(f"{INFO}would place {entry.name}  ->  {target_rel}")
                placed[target_rel] = target
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry, target)
            note = "" if entry.name == Path(target_rel).name else f"   (from {entry.name})"
            print(f"{OK}{target_rel}{note}")
            placed[target_rel] = target

    return placed


def verify_files(project: Path) -> Tuple[List[str], List[str]]:
    """Return (missing_required, missing_optional)."""
    missing_req, missing_opt = [], []
    for rel, (required, marker) in EXPECTED.items():
        path = project / rel
        if not path.exists() or path.stat().st_size == 0:
            (missing_req if required else missing_opt).append(rel)
            continue
        try:
            body = path.read_text(errors="ignore")
            if marker.lower() not in body.lower():
                print(f"{WARN}{rel} is present but looks wrong (expected to find '{marker}')")
        except Exception:
            pass
    return missing_req, missing_opt


# --------------------------------------------------------------------------- #
# Environment checks
# --------------------------------------------------------------------------- #
def check_python() -> bool:
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}.{sys.version_info[2]}"
    if (major, minor) in [(3, 10), (3, 11), (3, 12)]:
        print(f"{OK}Python {version}")
        return True
    if (major, minor) >= (3, 13):
        print(f"{BAD}Python {version} -- too new. LightGBM and pyarrow wheels lag a release.")
        print(f"       Install 3.11 and rebuild the venv:  python3.11 -m venv .venv")
        return False
    print(f"{BAD}Python {version} -- too old. Install 3.11 from python.org.")
    return False


def check_venv() -> bool:
    active = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if active:
        print(f"{OK}Virtual environment active  ({Path(sys.prefix).name})")
        return True
    print(f"{WARN}No virtual environment active.")
    if os.name == "nt":
        print("       python -m venv .venv  &&  .venv\\Scripts\\Activate.ps1")
    else:
        print("       python3 -m venv .venv  &&  source .venv/bin/activate")
    return False


def check_packages() -> List[str]:
    import importlib.util

    missing = []
    for module, package in RUNTIME_IMPORTS:
        if importlib.util.find_spec(module) is None:
            missing.append(package)
    if missing:
        print(f"{WARN}{len(missing)} package(s) not installed: {', '.join(missing)}")
        print("       pip install -r requirements.txt")
    else:
        print(f"{OK}All {len(RUNTIME_IMPORTS)} runtime packages installed")
    return missing


def check_api_key() -> bool:
    if os.getenv("ANTHROPIC_API_KEY", "").startswith("sk-ant-"):
        print(f"{OK}ANTHROPIC_API_KEY found in the environment")
        return True
    if (Path.cwd() / ".streamlit" / "secrets.toml").exists():
        print(f"{OK}.streamlit/secrets.toml exists")
        return True
    print(f"{WARN}No API key found. Either export ANTHROPIC_API_KEY, copy")
    print("       .streamlit/secrets.toml.example to .streamlit/secrets.toml,")
    print("       or just paste the key into the app sidebar at runtime.")
    return False


def check_imports(project: Path) -> bool:
    """Import the project's own modules -- catches truncated downloads."""
    sys.path.insert(0, str(project))
    try:
        import ingestion, notebook_builder  # noqa: F401
        print(f"{OK}Project modules import cleanly")
        return True
    except Exception as exc:
        print(f"{BAD}Project modules failed to import: {type(exc).__name__}: {exc}")
        print("       Usually a truncated or half-copied download. Re-download that file.")
        return False
    finally:
        sys.path.pop(0)



# --------------------------------------------------------------------------- #
# Desktop launcher
# --------------------------------------------------------------------------- #
MAC_LAUNCHER = """#!/bin/bash
# AutoDS-Agent launcher -- generated by bootstrap.py. Double-click to start.
PROJECT="__PROJECT__"
cd "$PROJECT" || { echo "Project folder not found: $PROJECT"; read -n 1 -s -r -p "Press any key to close."; exit 1; }

clear
echo "=============================="
echo "  Starting AutoDS-Agent"
echo "=============================="
echo

PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python is not installed. Opening the download page in your browser."
  echo "Install it, then double-click this launcher again."
  open "https://www.python.org/downloads/release/python-3119/" 2>/dev/null || true
  read -n 1 -s -r -p "Press any key to close."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "First run: building the environment. This takes 2-4 minutes..."
  "$PY" -m venv .venv || { echo "Could not create the environment."; read -n 1 -s -r -p "Press any key."; exit 1; }
fi
source .venv/bin/activate

if ! python -c "import streamlit, anthropic, pandas, sklearn, lightgbm" >/dev/null 2>&1; then
  echo "Installing libraries (first run only, 2-4 minutes)..."
  python -m pip install --upgrade pip --quiet
  python -m pip install -r requirements.txt || { echo "Install failed."; read -n 1 -s -r -p "Press any key."; exit 1; }
fi

echo
echo "Opening in your browser. Leave this window open while you work."
echo "To stop: close this window."
echo
streamlit run app.py
read -n 1 -s -r -p "AutoDS-Agent has stopped. Press any key to close."
"""

WIN_LAUNCHER = r"""@echo off
REM AutoDS-Agent launcher -- generated by bootstrap.py. Double-click to start.
title AutoDS-Agent
cd /d "__PROJECT__" || goto :nofolder

cls
echo ==============================
echo   Starting AutoDS-Agent
echo ==============================
echo.

where python >nul 2>&1
if errorlevel 1 goto :nopython

if not exist ".venv" (
  echo First run: building the environment. This takes 2-4 minutes...
  python -m venv .venv
  if errorlevel 1 goto :venvfail
)
call .venv\Scripts\activate.bat

python -c "import streamlit, anthropic, pandas, sklearn, lightgbm" >nul 2>&1
if errorlevel 1 (
  echo Installing libraries ^(first run only, 2-4 minutes^)...
  python -m pip install --upgrade pip --quiet
  python -m pip install -r requirements.txt
  if errorlevel 1 goto :pipfail
)

echo.
echo Opening in your browser. Leave this window open while you work.
echo To stop: close this window.
echo.
streamlit run app.py
pause
exit /b 0

:nopython
echo Python is not installed. Opening the download page in your browser.
echo Install it, tick "Add python.exe to PATH", then double-click this launcher again.
start https://www.python.org/downloads/release/python-3119/
pause
exit /b 1

:nofolder
echo Project folder not found: __PROJECT__
pause
exit /b 1

:venvfail
echo Could not create the environment.
pause
exit /b 1

:pipfail
echo Library install failed. Check your internet connection and try again.
pause
exit /b 1
"""


def _desktop_dir() -> Optional[Path]:
    """Find the real Desktop folder.

    On Windows, OneDrive Backup relocates the Desktop to
    C:/Users/<you>/OneDrive/Desktop, so ~/Desktop stops existing. The registry
    holds the authoritative path, so ask it first and fall back to the usual
    suspects (including localised OneDrive folder names).
    """
    if os.name == "nt":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
            )
            raw, _ = winreg.QueryValueEx(key, "Desktop")
            winreg.CloseKey(key)
            candidate = Path(os.path.expandvars(raw))
            if candidate.is_dir():
                return candidate
        except Exception:
            pass   # fall through to the candidate list

    home = Path.home()
    candidates: List[Path] = []
    if os.name == "nt":
        # OneDrive-redirected Desktop first: when OneDrive Backup is on, a leftover
        # ~/Desktop can still exist but is not the Desktop Explorer shows you.
        onedrive = os.getenv("OneDrive") or os.getenv("OneDriveConsumer") or ""
        if onedrive:
            candidates.append(Path(onedrive) / "Desktop")
        candidates += [p / "Desktop" for p in sorted(home.glob("OneDrive*")) if p.is_dir()]
        candidates.append(home / "Desktop")
        userprofile = os.getenv("USERPROFILE")
        if userprofile:
            candidates.append(Path(userprofile) / "Desktop")
    else:
        xdg = os.getenv("XDG_DESKTOP_DIR")
        if xdg:
            candidates.append(Path(os.path.expandvars(xdg)))
        candidates.append(home / "Desktop")

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def install_launcher(project: Path) -> None:
    """Write a double-click launcher into the project and onto the Desktop.

    Generated locally on purpose: a .command file downloaded from a browser loses
    its executable bit and opens in TextEdit instead of running.
    """
    windows = os.name == "nt"
    name = "Start AutoDS-Agent.bat" if windows else "Start AutoDS-Agent.command"
    body = (WIN_LAUNCHER if windows else MAC_LAUNCHER).replace("__PROJECT__", str(project))

    local = project / name
    local.write_text(body, newline="\r\n" if windows else "\n")
    if not windows:
        local.chmod(0o755)
    print(f"{OK}created {name}")

    desktop = _desktop_dir()
    if desktop is not None:
        shortcut = desktop / name
        try:
            shutil.copy2(local, shortcut)
            if not windows:
                shortcut.chmod(0o755)
            print(f"{OK}copied to your Desktop  ({desktop})")
        except Exception as exc:
            print(f"{WARN}could not copy to Desktop ({exc}); use the one in the project folder")
    else:
        print(f"{WARN}could not locate your Desktop folder.")
        print(f"       Drag this file to your Desktop yourself:")
        print(f"       {local}")

    print(f"\n  Double-click \"{name}\" to start the tool. You never need the terminal again.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble and verify an AutoDS-Agent project folder.")
    parser.add_argument("--from", dest="source", help="Folder holding the downloaded files")
    parser.add_argument("--into", dest="project", default=".", help="Project folder (default: here)")
    parser.add_argument("--check", action="store_true", help="Verify only; do not move anything")
    parser.add_argument("--install-launcher", action="store_true", help="Create a double-click launcher on your Desktop")
    args = parser.parse_args()

    project = Path(args.project).expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)

    if args.install_launcher:
        print("=" * 60)
        print("  AutoDS-Agent :: Desktop launcher")
        print("=" * 60)
        install_launcher(project)
        return 0

    print("=" * 60)
    print("  AutoDS-Agent :: Bootstrap")
    print("=" * 60)
    print(f"  Project folder: {project}")

    if not args.check:
        head("1. Placing files")
        sources = candidate_sources(args.source)
        print(f"  Looking in: {', '.join(str(s) for s in sources if s.is_dir())}\n")
        placed = collect(project, sources, dry_run=False)
        if not placed:
            print(f"{WARN}Nothing found. Download the files first, or point at them:")
            print("       python bootstrap.py --from ~/Downloads")

    head("2. Checking files")
    missing_req, missing_opt = verify_files(project)
    for rel in missing_req:
        print(f"{BAD}MISSING (required): {rel}")
    for rel in missing_opt:
        print(f"{WARN}missing (optional): {rel}")
    if not missing_req:
        print(f"{OK}All {len(CORE)} core modules and the required support files are present")

    head("3. Checking your environment")
    py_ok = check_python()
    check_venv()
    missing_pkgs = check_packages()
    check_api_key()

    imports_ok = True
    if not missing_req and not missing_pkgs:
        imports_ok = check_imports(project)

    head("Next step")
    if missing_req:
        print("  Download the missing file(s) above, then run this script again.")
        return 1
    if not py_ok:
        print("  Install Python 3.11, recreate the virtual environment, then re-run.")
        return 1
    if missing_pkgs:
        print("  Install the dependencies, then re-run this script:")
        print("      pip install -r requirements.txt")
        print("      python bootstrap.py --check")
        return 1
    if not imports_ok:
        print("  Re-download the module that failed to import, then re-run.")
        return 1

    print("  Everything checks out. Start the app:")
    print("      streamlit run app.py")
    print("\n  Then follow Part 4 of BUILD.md to make your first test dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
