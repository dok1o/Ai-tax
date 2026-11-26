"""Tests for the offline tokenizer backends (T-016).

The rule under test: the app either loads a real tokenizer from a local cache or
reports why it cannot. It never downloads, and it never quietly substitutes word
splitting for tokenization.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from app import token_backends

ENCODING, REPO, PINNED = token_backends.pinned_identifiers()
REVISION = PINNED.split("@")[-1]


def block_network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Tripwire every socket and HTTP entry point; return the recorded attempts."""
    import socket
    import urllib.request

    attempts: list[str] = []

    def tripwire(name: str):
        def boom(*_args: object, **_kwargs: object) -> None:
            attempts.append(name)
            raise AssertionError(f"network call: {name}")

        return boom

    monkeypatch.setattr(urllib.request, "urlopen", tripwire("urllib.request.urlopen"))
    monkeypatch.setattr(
        socket, "create_connection", tripwire("socket.create_connection")
    )
    monkeypatch.setattr(socket, "getaddrinfo", tripwire("socket.getaddrinfo"))
    monkeypatch.setattr(socket.socket, "connect", tripwire("socket.socket.connect"))
    requests = pytest.importorskip("requests")
    monkeypatch.setattr(requests, "get", tripwire("requests.get"))
    monkeypatch.setattr(
        requests.Session, "request", tripwire("requests.Session.request")
    )
    return attempts


def write_cache(directory: Path, blobpath: str, data: bytes) -> Path:
    """Put ``data`` where tiktoken would look for ``blobpath``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / hashlib.sha1(blobpath.encode()).hexdigest()
    path.write_bytes(data)
    return path


def test_pinned_identifiers_come_from_the_t001_module() -> None:
    """The ast read must agree with the real module, whatever it costs to import."""
    pytest.importorskip("transformers")
    from src import tokens

    assert ENCODING == tokens.TIKTOKEN_ENCODING
    assert REPO == tokens.SUBWORD_TOKENIZER_REPO
    assert PINNED == tokens.SUBWORD_TOKENIZER_ID
    assert PINNED.split("@")[-1] == tokens.SUBWORD_TOKENIZER_REVISION


def test_cache_locations_follow_the_documented_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "tiktoken"))
    assert token_backends.tiktoken_cache_dir() == tmp_path / "tiktoken"
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR")
    monkeypatch.delenv("DATA_GYM_CACHE_DIR", raising=False)
    assert token_backends.tiktoken_cache_dir().name == "data-gym-cache"
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "")
    assert token_backends.tiktoken_cache_dir() is None, "empty disables caching"

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    (tmp_path / "hf").mkdir()
    assert (
        token_backends.hf_cache_dir(REPO)
        == tmp_path / "hf" / f"models--{REPO.replace('/', '--')}"
    )
    (tmp_path / "hf" / "hub").mkdir()
    assert token_backends.hf_cache_dir(REPO).parent == tmp_path / "hf" / "hub"


def test_the_registry_spec_is_read_without_calling_it(monkeypatch) -> None:
    """Calling the registry function downloads; parsing its source does not."""
    attempts = block_network(monkeypatch)
    blobpath, expected = token_backends.tiktoken_cache_spec(ENCODING)
    assert blobpath.startswith("https://") and blobpath.endswith(".tiktoken")
    assert expected is not None and len(expected) == 64
    assert attempts == []


def test_only_a_readable_matching_cache_file_counts_as_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "cache"))
    blobpath, expected = token_backends.tiktoken_cache_spec(ENCODING)
    assert not token_backends.tiktoken_is_cached(ENCODING)
    assert "is not in" in token_backends.tiktoken_cache_state(ENCODING).reason

    corrupt = write_cache(tmp_path / "cache", blobpath, b"truncated bpe")
    state = token_backends.tiktoken_cache_state(ENCODING)
    assert not state.ok and "corrupt" in state.reason
    assert corrupt.is_file(), "the check must not delete what it rejects"

    write_cache(tmp_path / "cache", blobpath, b"")
    assert "empty" in token_backends.tiktoken_cache_state(ENCODING).reason

    # A file whose hash matches is the only thing that may be loaded.
    monkeypatch.setattr(
        token_backends,
        "tiktoken_cache_spec",
        lambda _encoding: (blobpath, hashlib.sha256(b"good bpe").hexdigest()),
    )
    write_cache(tmp_path / "cache", blobpath, b"good bpe")
    assert token_backends.tiktoken_is_cached(ENCODING)
    del expected


def test_an_unverifiable_hash_refuses_to_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No published hash means no offline proof, so the file stays unloaded."""
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    blobpath, _ = token_backends.tiktoken_cache_spec(ENCODING)
    write_cache(tmp_path, blobpath, b"some bpe")
    monkeypatch.setattr(
        token_backends, "tiktoken_cache_spec", lambda _encoding: (blobpath, None)
    )
    state = token_backends.tiktoken_cache_state(ENCODING)
    assert not state.ok and "no expected hash" in state.reason


def test_an_empty_cache_dir_is_reported_as_a_download_risk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "")
    state = token_backends.tiktoken_cache_state(ENCODING)
    assert not state.ok and "empty" in state.reason and "download" in state.reason


def test_corrupt_caches_never_open_a_socket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: a corrupt cache used to make tiktoken re-download it."""
    attempts = block_network(monkeypatch)
    blobpath, _ = token_backends.tiktoken_cache_spec(ENCODING)
    cache_dir = tmp_path / "tiktoken"
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(cache_dir))
    corrupt = write_cache(cache_dir, blobpath, b"corrupt bpe payload\n" * 10)
    refs = tmp_path / "hf" / "hub" / f"models--{REPO.replace('/', '--')}" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(REVISION, encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    report = token_backends.load_backends()
    assert attempts == [], "loading a corrupt cache must not touch the network"
    assert report.backends == {}
    assert "corrupt" in report.problems[ENCODING]
    assert "sha256" in report.problems[ENCODING]
    assert "refusing to download" in report.problems[ENCODING]
    assert corrupt.is_file(), "the corrupt file is reported, not deleted"
    assert "no snapshot" in report.problems[PINNED]
    assert "never downloads" in report.problems[PINNED]


def test_an_incomplete_snapshot_is_reported_before_the_loader_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempts = block_network(monkeypatch)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    snapshot = token_backends.hf_snapshot_dir(REPO, REVISION)
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer.json").write_bytes(b"")

    def refuse(*_args: object, **_kwargs: object) -> token_backends.Backend:
        raise AssertionError("the loader ran on an incomplete snapshot")

    monkeypatch.setattr(token_backends, "_subword_backend", refuse)
    report = token_backends.load_backends()
    assert attempts == []
    assert "incomplete" in report.problems[PINNED]
    assert PINNED not in report.identifiers


def test_the_tiktoken_loader_refuses_to_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Even without the pre-flight check, the loader itself cannot download."""
    tiktoken_registry = pytest.importorskip("tiktoken.registry")
    attempts = block_network(monkeypatch)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "cold"))
    # tiktoken keeps every loaded encoding in memory; a warm process would
    # return it without reading the cache. monkeypatch restores the original.
    monkeypatch.setattr(tiktoken_registry, "ENCODINGS", {})
    with pytest.raises(token_backends.OfflineCacheError):
        token_backends._tiktoken_backend(ENCODING)
    assert attempts == []


def test_the_subword_loader_never_connects_on_a_cold_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``local_files_only=True`` alone still opens a socket; offline mode does not."""
    pytest.importorskip("transformers")
    attempts = block_network(monkeypatch)
    # huggingface_hub reads HF_HOME at import, so a late setenv would still load
    # a warm cache; an explicit empty cache folder keeps the test cold.
    cold = tmp_path / "hf" / "hub"
    # The raised type varies with the transformers/hub version; what matters is
    # that it is raised here instead of on a socket.
    with pytest.raises(Exception) as failure:
        token_backends._subword_backend(REPO, REVISION, PINNED, cache_dir=cold)
    assert attempts == [], "from_pretrained must not connect to huggingface.co"
    assert (
        "offline" in str(failure.value).lower()
        or "connect" in str(failure.value).lower()
    )


def test_missing_caches_are_reported_and_no_loader_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))

    def refuse(*_args: object, **_kwargs: object) -> token_backends.Backend:
        raise AssertionError("a loader ran although the cache is missing")

    monkeypatch.setattr(token_backends, "_tiktoken_backend", refuse)
    monkeypatch.setattr(token_backends, "_subword_backend", refuse)
    before = "transformers" in sys.modules

    report = token_backends.load_backends()
    assert report.backends == {}
    assert set(report.problems) == {ENCODING, PINNED}
    assert "not in" in report.problems[ENCODING]
    assert "never downloads" in report.problems[PINNED]
    assert ("transformers" in sys.modules) == before


def test_warm_caches_load_through_the_pinned_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warm = token_backends.CacheCheck(True, "offline cache")
    monkeypatch.setattr(token_backends, "tiktoken_cache_state", lambda _encoding: warm)
    monkeypatch.setattr(
        token_backends, "hf_cache_state", lambda _repo, _revision, _identifier: warm
    )
    calls: list[tuple[str, ...]] = []

    def fake_tiktoken(encoding: str) -> token_backends.Backend:
        calls.append(("tiktoken", encoding))
        return token_backends.Backend(encoding, lambda text: [(1, text)])

    def fake_subword(
        repo: str, revision: str, identifier: str
    ) -> token_backends.Backend:
        calls.append(("hf", repo, revision, identifier))
        return token_backends.Backend(identifier, lambda text: [(2, text)])

    monkeypatch.setattr(token_backends, "_tiktoken_backend", fake_tiktoken)
    monkeypatch.setattr(token_backends, "_subword_backend", fake_subword)
    report = token_backends.load_backends()
    assert report.identifiers == [ENCODING, PINNED]
    assert report.problems == {}
    assert calls[0] == ("tiktoken", ENCODING)
    assert calls[1] == ("hf", REPO, PINNED.split("@")[-1], PINNED)


def test_a_backend_that_raises_is_reported_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warm = token_backends.CacheCheck(True, "offline cache")
    monkeypatch.setattr(token_backends, "tiktoken_cache_state", lambda _encoding: warm)
    monkeypatch.setattr(
        token_backends, "hf_cache_state", lambda _repo, _revision, _identifier: warm
    )

    def broken(_encoding: str) -> token_backends.Backend:
        raise RuntimeError("cache is corrupt")

    monkeypatch.setattr(token_backends, "_tiktoken_backend", broken)
    monkeypatch.setattr(token_backends, "_subword_backend", broken)
    report = token_backends.load_backends()
    assert report.backends == {}
    assert "corrupt" in report.problems[ENCODING]


def test_word_split_is_positional_and_labelled() -> None:
    pieces = token_backends.word_split_pieces("Сәлем әлем!")
    assert pieces == [(0, "Сәлем"), (1, "әлем"), (2, "!")]
    backend = token_backends.word_split_backend()
    assert backend.identifier == token_backends.WORD_SPLIT_ID == "word-split"
    assert token_backends.WORD_SPLIT_LABEL == "words, not tokens"
    assert backend.pieces("  ") == []


def test_backend_pieces_never_invents_tokens() -> None:
    assert token_backends.backend_pieces("Сәлем әлем", None) == []
    assert token_backends.backend_pieces("Сәлем әлем", None, diagnostic=True) == [
        (0, "Сәлем"),
        (1, "әлем"),
    ]
    backend = token_backends.Backend("fake", lambda text: [(9, text)])
    assert token_backends.backend_pieces("Сәлем", backend) == [(9, "Сәлем")]
    assert token_backends.count([(9, "Сәлем")]) == 1


def test_tiktoken_backend_replaces_a_partial_byte_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A byte-level BPE splits multi-byte Kazakh letters; chips must still render."""
    import types

    class FakeEncoding:
        def encode(self, _text: str, disallowed_special: str = ()) -> list[int]:
            assert disallowed_special == (), "no text may be rejected as special"
            return [220, 1079]  # a lone continuation byte, then "қ"

        def decode(self, tokens: list[int], errors: str = "strict") -> str:
            raw = {220: b"\xdc", 1079: "қ".encode()}[tokens[0]]
            return raw.decode("utf-8", errors=errors)

    module = types.ModuleType("tiktoken")
    module.get_encoding = lambda _name: FakeEncoding()
    monkeypatch.setitem(sys.modules, "tiktoken", module)
    backend = token_backends._tiktoken_backend("cl100k_base")
    pieces = backend.pieces("Қазақ")
    assert [piece[0] for piece in pieces] == [220, 1079]
    assert pieces[0][1] == "\ufffd", "the partial byte is replaced, not fatal"
    assert pieces[1][1] == "қ"


SHADOW_PROBE = """
import sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import tokenizers
assert hasattr(tokenizers, "Encoding"), tokenizers.__file__
from app import token_backends
encoding, repo, pinned = token_backends.pinned_identifiers()
revision = pinned.split("@")[-1]
if token_backends.hf_cache_state(repo, revision, pinned).ok:
    backend = token_backends._subword_backend(repo, revision, pinned)
    assert backend.pieces("Сәлем")
    print("qwen-loaded")
print(tokenizers.__file__)
"""


def test_app_directory_first_on_sys_path_keeps_the_real_tokenizers_package() -> None:
    """Streamlit puts app/ first on sys.path; nothing there may shadow a library."""
    import os
    import subprocess

    root = Path(__file__).resolve().parents[2]
    assert not (root / "app" / "tokenizers.py").exists()
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", SHADOW_PROBE, str(root / "app"), str(root)],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=root / "app",
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert str(root / "app") not in result.stdout


def test_tiktoken_backend_exposes_raw_token_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    class FakeEncoding:
        def encode(self, _text: str, **_kwargs: object) -> list[int]:
            return [7]

        def decode(self, _tokens: list[int], errors: str = "strict") -> str:
            return b"\xd0".decode("utf-8", errors=errors)

        def decode_single_token_bytes(self, token: int) -> bytes:
            return {7: b"\xd0"}[token]

    module = types.ModuleType("tiktoken")
    module.get_encoding = lambda _name: FakeEncoding()
    monkeypatch.setitem(sys.modules, "tiktoken", module)
    backend = token_backends._tiktoken_backend("cl100k_base")
    assert backend.token_bytes is not None
    assert backend.token_bytes(7) == b"\xd0"


def test_byte_level_decoder_inverts_the_gpt2_table() -> None:
    decoder = token_backends.BYTE_LEVEL_DECODER
    assert len(decoder) == 256 and sorted(decoder.values()) == list(range(256))
    assert decoder["Ġ"] == 0x20, "the space marker of byte-level BPE"
    assert bytes(decoder[char] for char in "ÐŁ") == "П".encode()
