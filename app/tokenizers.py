"""Real offline tokenizer backends for the Playground (T-016).

Two backends matter, both pinned by ``src/tokens.py`` (T-001): tiktoken
``cl100k_base`` and ``Qwen/Qwen2.5-0.5B`` at one revision. Neither is ever
downloaded from the app: the cache is checked on disk first, and a backend whose
cache is missing is reported as unavailable instead of being replaced by a word
splitter. Chips therefore always show model tokens, or nothing at all.

The pinned identifiers are read from ``src/tokens.py`` with ``ast`` rather than
imported: importing that module pulls in ``transformers``, which costs seconds of
cold start and is optional at runtime. ``tests/arena/test_tokenizers.py`` pins the
extracted values against the real module.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.common.config import ROOT

TOKENS_MODULE = ROOT / "src" / "tokens.py"
WORD_SPLIT_ID = "word-split"
WORD_SPLIT_LABEL = "words, not tokens"
PIECES = re.compile(r"\w+|[^\w\s]", re.UNICODE)
# tiktoken's public registry URL for cl100k_base. It is used only to look for the
# cached BPE file on disk; this module never opens a connection.
CL100K_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
CACHE_ENVS = ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR")


@dataclass(frozen=True)
class Backend:
    """One usable tokenizer: an identifier and a text -> (id, piece) encoder."""

    identifier: str
    encode: Callable[[str], list[tuple[int, str]]]

    def pieces(self, text: str) -> list[tuple[int, str]]:
        """Token ids with the decoded string of each id, in reading order."""
        if not text.strip():
            return []
        return self.encode(text)


@dataclass(frozen=True)
class BackendReport:
    """What loaded, and why everything else is unavailable."""

    backends: dict[str, Backend]
    problems: dict[str, str]

    @property
    def identifiers(self) -> list[str]:
        return list(self.backends)

    def get(self, identifier: str) -> Backend | None:
        return self.backends.get(identifier)


def pinned_identifiers() -> tuple[str, str, str]:
    """(tiktoken encoding, subword repo, subword repo@revision) from T-001."""
    tree = ast.parse(TOKENS_MODULE.read_text(encoding="utf-8"))
    values: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except ValueError:
            continue
    encoding = values.get("TIKTOKEN_ENCODING", "cl100k_base")
    repo = values.get("SUBWORD_TOKENIZER_REPO", "")
    revision = values.get("SUBWORD_TOKENIZER_REVISION", "")
    pinned = values.get("SUBWORD_TOKENIZER_ID", "")
    if not pinned and repo and revision:
        pinned = f"{repo}@{revision}"
    return encoding, repo, pinned


def tiktoken_cache_dir() -> Path:
    """Directory tiktoken reads its cached BPE files from."""
    for name in CACHE_ENVS:
        value = os.environ.get(name)
        if value:
            return Path(value)
    return Path(tempfile.gettempdir()) / "data-gym-cache"


def tiktoken_is_cached(url: str = CL100K_URL) -> bool:
    """True when the BPE file for ``url`` is already on disk."""
    digest = hashlib.sha1(url.encode()).hexdigest()
    return (tiktoken_cache_dir() / digest).is_file()


def hf_cache_dir(repo: str) -> Path:
    """Hugging Face hub cache folder of one repository."""
    root = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    base = Path(root) if root else Path.home() / ".cache" / "huggingface"
    # HF_HOME points at the hub folder itself on some installs, at its parent on
    # others; accept either instead of guessing one layout.
    hub = base if root and not (base / "hub").is_dir() else base / "hub"
    return hub / f"models--{repo.replace('/', '--')}"


def _tiktoken_backend(encoding: str) -> Backend:
    """Load one tiktoken encoding from the local cache only."""
    import tiktoken

    handle = tiktoken.get_encoding(encoding)

    def encode(text: str) -> list[tuple[int, str]]:
        ids = handle.encode(text, disallowed_special=())
        # One id can hold half of a multi-byte character, which is normal for a
        # byte-level BPE: show the replacement character instead of raising.
        return [(int(token), handle.decode([token], errors="replace")) for token in ids]

    return Backend(identifier=encoding, encode=encode)


def _subword_backend(repo: str, revision: str, identifier: str) -> Backend:
    """Load the pinned subword tokenizer with ``local_files_only=True``."""
    from transformers import AutoTokenizer

    handle = AutoTokenizer.from_pretrained(
        repo, revision=revision, use_fast=True, local_files_only=True
    )

    def encode(text: str) -> list[tuple[int, str]]:
        ids = handle.encode(text, add_special_tokens=False)
        return [(int(token), handle.decode([token])) for token in ids]

    return Backend(identifier=identifier, encode=encode)


def load_backends() -> BackendReport:
    """Try both real backends offline and report every failure with a reason."""
    encoding, repo, pinned = pinned_identifiers()
    backends: dict[str, Backend] = {}
    problems: dict[str, str] = {}

    if tiktoken_is_cached():
        try:
            backends[encoding] = _tiktoken_backend(encoding)
        except Exception as error:  # noqa: BLE001 - reported, never raised
            problems[encoding] = f"{type(error).__name__}: {error}"
    else:
        problems[encoding] = (
            f"the {encoding} BPE file is not in {tiktoken_cache_dir()}; warm it once "
            "with `python run.py tokens` on a machine with network access"
        )

    if repo and hf_cache_dir(repo).is_dir():
        try:
            backends[pinned] = _subword_backend(repo, pinned.split("@")[-1], pinned)
        except Exception as error:  # noqa: BLE001 - reported, never raised
            problems[pinned] = f"{type(error).__name__}: {error}"
    else:
        problems[pinned or repo] = (
            f"{pinned or repo} is not in the Hugging Face cache "
            f"({hf_cache_dir(repo)}); the app "
            "never downloads, so this tokenizer stays disabled"
        )

    return BackendReport(backends=backends, problems=problems)


def word_split_pieces(text: str) -> list[tuple[int, str]]:
    """Diagnostic word splitting. Not a tokenizer: every id here is positional."""
    if not text.strip():
        return []
    return [(index, piece) for index, piece in enumerate(PIECES.findall(text))]


def word_split_backend() -> Backend:
    """The labelled diagnostic backend, used only when explicitly switched on."""
    return Backend(identifier=WORD_SPLIT_ID, encode=word_split_pieces)


def backend_pieces(
    text: str, backend: Backend | None, diagnostic: bool = False
) -> list[tuple[int, str]]:
    """Pieces of one phrase, or an empty list when no tokenizer is available."""
    if backend is not None:
        return backend.pieces(text)
    return word_split_pieces(text) if diagnostic else []


def count(pieces: Sequence[tuple[int, str]]) -> int:
    """Token count of one phrase."""
    return len(pieces)
