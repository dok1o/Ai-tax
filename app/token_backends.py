"""Real offline tokenizer backends for the Playground (T-016).

Two backends matter, both pinned by ``src/tokens.py`` (T-001): tiktoken
``cl100k_base`` and ``Qwen/Qwen2.5-0.5B`` at one revision. Neither is ever
downloaded from the app: every cache is checked on disk and verified before a
loader runs, both loaders run with their download path blocked, and a backend
whose cache is missing, incomplete or corrupt is reported as unavailable instead
of being replaced by a word splitter. Chips therefore always show model tokens,
or nothing at all.

The pre-flight check is not decoration. ``tiktoken.load.read_file_cached`` deletes
a cached file whose sha256 does not match and then downloads a fresh copy, and
``AutoTokenizer.from_pretrained(..., local_files_only=True)`` still opens a socket
to huggingface.co when the pinned snapshot is incomplete. Both are stopped here
before the library is called.

The pinned identifiers are read from ``src/tokens.py`` with ``ast`` rather than
imported: importing that module pulls in ``transformers``, which costs seconds of
cold start and is optional at runtime. ``tests/arena/test_token_backends.py`` pins the
extracted values against the real module.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
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
HF_OFFLINE_ENVS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
# Either file is enough to prove the snapshot is a usable tokenizer, not a stub.
HF_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


@dataclass(frozen=True)
class Backend:
    """One usable tokenizer: an identifier and a text -> (id, piece) encoder.

    ``token_bytes`` returns the raw bytes of one id, so a chip can show a token
    that holds only part of a multi-byte letter as bytes instead of "�".
    """

    identifier: str
    encode: Callable[[str], list[tuple[int, str]]]
    token_bytes: Callable[[int], bytes] | None = None

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


class OfflineCacheError(RuntimeError):
    """Raised instead of a download: this module never reaches the network."""


@dataclass(frozen=True)
class CacheCheck:
    """One offline pre-flight result: usable, and the reason it is not."""

    ok: bool
    reason: str


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


def tiktoken_cache_dir() -> Path | None:
    """Directory tiktoken reads cached BPE files from; ``None`` means no cache.

    Mirrors ``tiktoken.load.read_file_cached``: the first variable that is *set*
    wins even when it is empty, and an empty value switches caching off, which
    makes tiktoken download on every single call.
    """
    for name in CACHE_ENVS:
        if name in os.environ:
            return Path(os.environ[name]) if os.environ[name] else None
    return Path(tempfile.gettempdir()) / "data-gym-cache"


def _literal(node: ast.AST) -> object:
    """Value of a literal node, or ``None`` when it is not a literal."""
    try:
        return ast.literal_eval(node)
    except ValueError:
        return None


def tiktoken_cache_spec(encoding: str) -> tuple[str, str | None]:
    """Blobpath and expected sha256 of one encoding, read without calling it.

    Calling ``tiktoken_ext.openai_public.<encoding>()`` downloads the BPE file
    when the cache is cold, so the installed registry module is parsed instead.
    """
    try:
        import tiktoken_ext.openai_public as registry

        source = Path(registry.__file__).read_text(encoding="utf-8")
    except (ImportError, OSError):
        return CL100K_URL, None
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef) or node.name != encoding:
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call) or not call.args:
                continue
            if getattr(call.func, "id", "") != "load_tiktoken_bpe":
                continue
            blobpath = _literal(call.args[0])
            if not isinstance(blobpath, str):
                continue
            expected = None
            for keyword in call.keywords:
                if keyword.arg == "expected_hash":
                    value = _literal(keyword.value)
                    expected = value if isinstance(value, str) else None
            return blobpath, expected
    return CL100K_URL, None


def tiktoken_cache_state(encoding: str) -> CacheCheck:
    """Existence, readability and hash of the cached BPE file, all offline."""
    blobpath, expected = tiktoken_cache_spec(encoding)
    cache_dir = tiktoken_cache_dir()
    if cache_dir is None:
        return CacheCheck(
            False,
            f"{CACHE_ENVS[0]} is set but empty, which disables the tiktoken cache "
            "and would download on every load",
        )
    path = cache_dir / hashlib.sha1(blobpath.encode()).hexdigest()
    if not path.is_file():
        return CacheCheck(
            False,
            f"the {encoding} BPE file is not in {cache_dir}; warm it once with "
            "`python run.py tokens` on a machine with network access",
        )
    try:
        data = path.read_bytes()
    except OSError as error:
        return CacheCheck(False, f"{path} is not readable: {error}")
    if not data:
        return CacheCheck(False, f"{path} is empty; refusing to download a copy")
    if expected is None:
        return CacheCheck(
            False,
            f"the {encoding} registry publishes no expected hash, so the cached "
            "file cannot be verified offline; refusing to load it",
        )
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        return CacheCheck(
            False,
            f"{path} is corrupt: sha256 {actual[:12]}… instead of {expected[:12]}…; "
            "refusing to download a copy",
        )
    return CacheCheck(True, "offline cache")


def tiktoken_is_cached(encoding: str = "cl100k_base") -> bool:
    """True when the cached BPE file is present, readable and not corrupt."""
    return tiktoken_cache_state(encoding).ok


def hf_cache_dir(repo: str) -> Path:
    """Hugging Face hub cache folder of one repository."""
    root = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    base = Path(root) if root else Path.home() / ".cache" / "huggingface"
    # HF_HOME points at the hub folder itself on some installs, at its parent on
    # others; accept either instead of guessing one layout.
    hub = base if root and not (base / "hub").is_dir() else base / "hub"
    return hub / f"models--{repo.replace('/', '--')}"


def hf_snapshot_dir(repo: str, revision: str) -> Path:
    """Directory the hub reads one pinned revision from."""
    return hf_cache_dir(repo) / "snapshots" / revision


def hf_cache_state(repo: str, revision: str, identifier: str) -> CacheCheck:
    """A pinned snapshot holding a readable tokenizer file, checked offline."""
    if not repo or not revision:
        return CacheCheck(False, f"{identifier or repo} is not pinned in src/tokens.py")
    snapshot = hf_snapshot_dir(repo, revision)
    if not snapshot.is_dir():
        return CacheCheck(
            False,
            f"{identifier} has no snapshot in the Hugging Face cache ({snapshot}); "
            "the app never downloads, so this tokenizer stays disabled",
        )
    for name in HF_TOKENIZER_FILES:
        path = snapshot / name
        try:
            if not path.is_file():
                continue
            with path.open("rb") as handle:
                if handle.read(1):
                    return CacheCheck(True, "offline cache")
        except OSError:
            continue
    return CacheCheck(
        False,
        f"{snapshot} holds no readable {'/'.join(HF_TOKENIZER_FILES)}: the "
        "snapshot is incomplete and the app never downloads",
    )


def _byte_level_decoder() -> dict[str, int]:
    """Inverse of the GPT-2 byte-to-unicode table of byte-level BPE vocabularies."""
    printable = [
        *range(ord("!"), ord("~") + 1),
        *range(ord("¡"), ord("¬") + 1),
        *range(ord("®"), ord("ÿ") + 1),
    ]
    codes = list(printable)
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codes.append(256 + extra)
            extra += 1
    return {chr(code): byte for byte, code in zip(printable, codes, strict=True)}


BYTE_LEVEL_DECODER = _byte_level_decoder()


def _refuse_download(*_args: object, **_kwargs: object) -> bytes:
    """Tripwire: every tiktoken download ends here instead of on a socket."""
    raise OfflineCacheError("the cache is unusable and the app never downloads")


@contextmanager
def _blocked_downloads() -> Iterator[None]:
    """Block tiktoken's only download entry point for the length of one load."""
    import tiktoken.load as tiktoken_load

    original = tiktoken_load.read_file
    tiktoken_load.read_file = _refuse_download
    try:
        yield
    finally:
        tiktoken_load.read_file = original


@contextmanager
def _hub_offline() -> Iterator[None]:
    """Force huggingface_hub offline; ``local_files_only`` alone still connects."""
    saved = {name: os.environ.get(name) for name in HF_OFFLINE_ENVS}
    for name in HF_OFFLINE_ENVS:
        os.environ[name] = "1"
    patched = False
    previous = False
    try:
        from huggingface_hub import constants

        previous = bool(constants.HF_HUB_OFFLINE)
        constants.HF_HUB_OFFLINE = True
        patched = True
        yield
    finally:
        if patched:
            constants.HF_HUB_OFFLINE = previous
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _tiktoken_backend(encoding: str) -> Backend:
    """Load one tiktoken encoding from a verified local cache, never the network."""
    import tiktoken

    with _blocked_downloads():
        handle = tiktoken.get_encoding(encoding)

    def encode(text: str) -> list[tuple[int, str]]:
        ids = handle.encode(text, disallowed_special=())
        # One id can hold half of a multi-byte character, which is normal for a
        # byte-level BPE: show the replacement character instead of raising.
        return [(int(token), handle.decode([token], errors="replace")) for token in ids]

    def token_bytes(token: int) -> bytes:
        return handle.decode_single_token_bytes(token)

    return Backend(identifier=encoding, encode=encode, token_bytes=token_bytes)


def _subword_backend(
    repo: str, revision: str, identifier: str, cache_dir: Path | None = None
) -> Backend:
    """Load the pinned subword tokenizer offline: local files plus hub offline.

    ``cache_dir`` defaults to the hub cache that ``huggingface_hub`` resolved at
    import time; tests pass an explicit folder because that import-time value
    ignores a later ``HF_HOME``.
    """
    from transformers import AutoTokenizer

    with _hub_offline():
        handle = AutoTokenizer.from_pretrained(
            repo,
            revision=revision,
            use_fast=True,
            local_files_only=True,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )

    def encode(text: str) -> list[tuple[int, str]]:
        ids = handle.encode(text, add_special_tokens=False)
        return [(int(token), handle.decode([token])) for token in ids]

    def token_bytes(token: int) -> bytes:
        piece = handle.convert_ids_to_tokens(token)
        return bytes(BYTE_LEVEL_DECODER[char] for char in piece)

    return Backend(identifier=identifier, encode=encode, token_bytes=token_bytes)


def load_backends() -> BackendReport:
    """Try both real backends offline and report every failure with a reason."""
    encoding, repo, pinned = pinned_identifiers()
    backends: dict[str, Backend] = {}
    problems: dict[str, str] = {}

    cache = tiktoken_cache_state(encoding)
    if cache.ok:
        try:
            backends[encoding] = _tiktoken_backend(encoding)
        except Exception as error:  # noqa: BLE001 - reported, never raised
            problems[encoding] = f"{type(error).__name__}: {error}"
    else:
        problems[encoding] = cache.reason

    revision = pinned.split("@")[-1] if "@" in pinned else ""
    snapshot = hf_cache_state(repo, revision, pinned or repo)
    if snapshot.ok:
        try:
            backends[pinned] = _subword_backend(repo, revision, pinned)
        except Exception as error:  # noqa: BLE001 - reported, never raised
            problems[pinned] = f"{type(error).__name__}: {error}"
    else:
        problems[pinned or repo] = snapshot.reason

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
