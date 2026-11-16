"""Program workbench: look at, edit and re-run the project DataPilot generated.

The project is the same zip the app offers for download, held as {path: text or bytes}.
Edits stay in memory until the user downloads the project or runs a script, which is
executed with the current Python in a temporary directory next to a copy of the data."""
from __future__ import annotations

import io
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = "datapilot_project/"
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".json", ".cfg", ".csv"}


def _is_text(name: str) -> bool:
    return PurePosixPath(name).suffix.lower() in TEXT_SUFFIXES


def load_project(zip_bytes: bytes) -> dict[str, str | bytes]:
    """Zip bytes -> {relative path: str for text files, bytes for binaries}."""
    files: dict[str, str | bytes] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename.removeprefix(ROOT)
            data = z.read(info)
            files[name] = data.decode("utf-8") if _is_text(name) else data
    return files


def dump_project(files: dict[str, str | bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(ROOT + name, content if isinstance(content, bytes) else content.encode("utf-8"))
    return buf.getvalue()


def ordered_names(files: dict[str, str | bytes]) -> list[str]:
    """Scripts you are meant to edit first, then docs, then the library, then binaries."""
    def key(n: str):
        if n in ("train.py", "make_submission.py", "predict.py"):
            return (0, n)
        if n in ("README.md", "requirements.txt"):
            return (1, n)
        if n.startswith("datapilot/"):
            return (2, n)
        return (3, n)
    return sorted(files, key=key)


def changed_files(original: dict, current: dict) -> list[str]:
    return [n for n in current if n in original and original[n] != current[n]]


def run_script_files(files: dict[str, str | bytes], script: str, data_csv: bytes | None,
                     extra: dict[str, bytes] | None = None, collect: tuple[str, ...] = (),
                     timeout: int = 1800) -> tuple[int, str, str, dict[str, bytes]]:
    """Runs `python <script> data.csv` inside a scratch copy of the project.

    `extra` are more input files placed next to it (test.csv, sample_submission.csv);
    `collect` names output files to read back. Returns (exit code, stdout, stderr,
    {collected name: bytes}); exit code -1 means it timed out."""
    if script not in files or not script.endswith(".py"):
        raise ValueError(f"{script!r} is not a Python file in this project")
    with tempfile.TemporaryDirectory(prefix="datapilot_run_") as tmp:
        root = Path(tmp).resolve()
        for name, content in {**files, **(extra or {})}.items():
            dest = (root / name).resolve()
            if root not in dest.parents:
                raise ValueError(f"refusing to write outside the project: {name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
        args = [sys.executable, script]
        if data_csv is not None:
            (root / "data.csv").write_bytes(data_csv)
            args.append("data.csv")
        try:
            r = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as e:
            return -1, (e.stdout or ""), f"Stopped after {timeout}s.", {}
        got = {n: (root / n).read_bytes() for n in collect if (root / n).is_file()}
        return r.returncode, r.stdout, r.stderr, got


def run_script(files: dict[str, str | bytes], script: str, data_csv: bytes | None,
               timeout: int = 1800) -> tuple[int, str, str]:
    """Same as run_script_files without extra inputs or collected outputs."""
    code, out, err, _ = run_script_files(files, script, data_csv, timeout=timeout)
    return code, out, err


def split_sections(src: str) -> tuple[str, list[tuple[str, str]]]:
    """A script written as '# %% Title' blocks -> (text before the first block, [(title, body)])."""
    parts = re.split(r"(?m)^# %% ?", src)
    head, blocks = parts[0], []
    for p in parts[1:]:
        title, _, body = p.partition("\n")
        blocks.append((title.strip(), body))
    return head, blocks


def join_sections(head: str, blocks: list[tuple[str, str]]) -> str:
    return head + "".join(f"# %% {t}\n{b}" for t, b in blocks)
