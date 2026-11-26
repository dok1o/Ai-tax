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

from app import tokenizers

ENCODING, REPO, PINNED = tokenizers.pinned_identifiers()


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
    assert tokenizers.tiktoken_cache_dir() == tmp_path / "tiktoken"
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR")
    monkeypatch.delenv("DATA_GYM_CACHE_DIR", raising=False)
    assert tokenizers.tiktoken_cache_dir().name == "data-gym-cache"

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    (tmp_path / "hf").mkdir()
    assert (
        tokenizers.hf_cache_dir(REPO)
        == tmp_path / "hf" / f"models--{REPO.replace('/', '--')}"
    )
    (tmp_path / "hf" / "hub").mkdir()
    assert tokenizers.hf_cache_dir(REPO).parent == tmp_path / "hf" / "hub"


def test_tiktoken_cache_detection_is_a_file_check(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    assert not tokenizers.tiktoken_is_cached()
    digest = hashlib.sha1(tokenizers.CL100K_URL.encode()).hexdigest()
    (tmp_path / digest).write_bytes(b"cached bpe")
    assert tokenizers.tiktoken_is_cached()


def test_missing_caches_are_reported_and_no_loader_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))

    def refuse(*_args: object, **_kwargs: object) -> tokenizers.Backend:
        raise AssertionError("a loader ran although the cache is missing")

    monkeypatch.setattr(tokenizers, "_tiktoken_backend", refuse)
    monkeypatch.setattr(tokenizers, "_subword_backend", refuse)
    before = "transformers" in sys.modules

    report = tokenizers.load_backends()
    assert report.backends == {}
    assert set(report.problems) == {ENCODING, PINNED}
    assert "not in" in report.problems[ENCODING]
    assert "never downloads" in report.problems[PINNED]
    assert ("transformers" in sys.modules) == before


def test_warm_caches_load_through_the_pinned_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(tokenizers, "tiktoken_is_cached", lambda _url=None: True)
    (tmp_path / "hf" / "hub" / f"models--{REPO.replace('/', '--')}").mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    calls: list[tuple[str, ...]] = []

    def fake_tiktoken(encoding: str) -> tokenizers.Backend:
        calls.append(("tiktoken", encoding))
        return tokenizers.Backend(encoding, lambda text: [(1, text)])

    def fake_subword(repo: str, revision: str, identifier: str) -> tokenizers.Backend:
        calls.append(("hf", repo, revision, identifier))
        return tokenizers.Backend(identifier, lambda text: [(2, text)])

    monkeypatch.setattr(tokenizers, "_tiktoken_backend", fake_tiktoken)
    monkeypatch.setattr(tokenizers, "_subword_backend", fake_subword)
    report = tokenizers.load_backends()
    assert report.identifiers == [ENCODING, PINNED]
    assert report.problems == {}
    assert calls[0] == ("tiktoken", ENCODING)
    assert calls[1] == ("hf", REPO, PINNED.split("@")[-1], PINNED)


def test_a_backend_that_raises_is_reported_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tokenizers, "tiktoken_is_cached", lambda _url=None: True)
    monkeypatch.setattr(tokenizers, "hf_cache_dir", lambda _repo: Path("/nonexistent"))

    def broken(_encoding: str) -> tokenizers.Backend:
        raise RuntimeError("cache is corrupt")

    monkeypatch.setattr(tokenizers, "_tiktoken_backend", broken)
    report = tokenizers.load_backends()
    assert report.backends == {}
    assert "corrupt" in report.problems[ENCODING]


def test_word_split_is_positional_and_labelled() -> None:
    pieces = tokenizers.word_split_pieces("Сәлем әлем!")
    assert pieces == [(0, "Сәлем"), (1, "әлем"), (2, "!")]
    backend = tokenizers.word_split_backend()
    assert backend.identifier == tokenizers.WORD_SPLIT_ID == "word-split"
    assert tokenizers.WORD_SPLIT_LABEL == "words, not tokens"
    assert backend.pieces("  ") == []


def test_backend_pieces_never_invents_tokens() -> None:
    assert tokenizers.backend_pieces("Сәлем әлем", None) == []
    assert tokenizers.backend_pieces("Сәлем әлем", None, diagnostic=True) == [
        (0, "Сәлем"),
        (1, "әлем"),
    ]
    backend = tokenizers.Backend("fake", lambda text: [(9, text)])
    assert tokenizers.backend_pieces("Сәлем", backend) == [(9, "Сәлем")]
    assert tokenizers.count([(9, "Сәлем")]) == 1


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
    backend = tokenizers._tiktoken_backend("cl100k_base")
    pieces = backend.pieces("Қазақ")
    assert [piece[0] for piece in pieces] == [220, 1079]
    assert pieces[0][1] == "\ufffd", "the partial byte is replaced, not fatal"
    assert pieces[1][1] == "қ"
