"""Tests for the shared ``ru_lemma`` lemmatizing tokenizer (Story 3.1).

Pure, no network, no real DB file — an in-memory ``apsw.Connection(":memory:")``
suffices (the tokenizer never touches the on-disk store). Tests assert the REAL
guarantee — the actual lemma lists and the actual FTS5 ``MATCH`` counts — never a
"did it register" flag (per the behavior-first review lesson carried 1.1 → 2.4).

Expected lemmas are pinned to what ``pymorphy3==2.0.6`` +
``pymorphy3-dicts-ru==2.4.417150.4580142`` actually return (verified this run
against the pinned dict), not from memory.
"""

from __future__ import annotations

import apsw
import pytest

from tgkb.common import ru_lemma


def _conn() -> apsw.Connection:
    """A throwaway in-memory connection — no on-disk store needed."""
    return apsw.Connection(":memory:")


def test_lemmatize_inflections_share_lemma() -> None:
    """AC#4: inflected forms all reduce to the common lemma ``договор``.

    This shared lemma is the mechanism that later makes them mutually matchable.
    """
    for form in ("договоры", "договора", "договором", "договор"):
        assert "договор" in ru_lemma.lemmatize(form), form


def test_lemmatize_emits_all_homonym_hypotheses() -> None:
    """AC#2: a homonym emits ALL ranked hypotheses (recall favored), deduped.

    ``стали`` is genuinely ambiguous — verb ``стать`` (best score) and noun
    ``сталь`` — and pymorphy3 returns ``сталь`` several times. ``lemmatize``
    keeps BOTH distinct lemmas, in rank order, with no duplicates.
    """
    lemmas = ru_lemma.lemmatize("стали")
    assert "стать" in lemmas
    assert "сталь" in lemmas
    # Recall-favored: verb hypothesis (best score) comes first.
    assert lemmas.index("стать") < lemmas.index("сталь")
    # No duplicate lemma bloating the index.
    assert len(lemmas) == len(set(lemmas))


def test_lemmatize_dedups_repeated_normal_forms() -> None:
    """``договоры`` — pymorphy3 emits ``договор`` twice → one deduped lemma."""
    assert ru_lemma.lemmatize("договоры") == ["договор"]


def test_lemmatize_fallback_on_unparseable() -> None:
    """A non-word returns the surface form ``[token]``, never ``[]``.

    ``TransformTokenizer`` treats ``[]`` as "remove the token"; silently
    dropping a token is a latent recall hole. The surface form keeps it at least
    literally matchable.
    """
    token = "zzqxwv"
    lemmas = ru_lemma.lemmatize(token)
    assert lemmas != []
    assert token in lemmas


def test_lemmatize_never_emits_blank_token() -> None:
    """Blank input never produces a blank searchable term.

    Directly exercises the fallback branch: an empty / whitespace token yields
    no meaningful lemma, and ``lemmatize`` must emit nothing rather than index
    ``""`` (a garbage FTS term). The base ``unicode61`` tokenizer never hands the
    transform an empty token, but ``lemmatize`` is public, so guard it directly.
    """
    assert ru_lemma.lemmatize("") == []
    # Any result must never contain a blank term for whitespace either.
    assert "" not in ru_lemma.lemmatize("   ")


def test_tokenizer_registers_and_tokenizes_cyrillic() -> None:
    """AC#1, #3: the ``TransformTokenizer(unicode61)`` chain segments + lemmatizes.

    A mixed Cyrillic+Latin string: Cyrillic words are segmented and lemmatized
    (not dropped), Latin words are folded by ``unicode61``. The homonym shares
    one position as a co-located tuple.
    """
    positions = ru_lemma.tokenize(_conn(), "договоры стали Hello")
    # One position per source word (co-located homonym stays one position).
    assert positions == [("договор",), ("стать", "сталь"), ("hello",)]


def test_end_to_end_match_via_fts5() -> None:
    """AC#4 payoff: index-side and query-side lemmatization AGREE by construction.

    Create an in-memory FTS5 table with ``tokenize="ru_lemma unicode61"``, index
    a row containing ``договоры``, and confirm a query for a DIFFERENT inflection
    (``договор`` / ``договором``) still matches. This mirrors, in-test, exactly
    what Story 3.2 will do for real — a behavioral proof of THIS story's object.
    """
    conn = _conn()
    ru_lemma.register(conn)
    conn.execute(
        f'CREATE VIRTUAL TABLE t USING fts5(body, tokenize="{ru_lemma.TOKENIZER_NAME} {ru_lemma.BASE_TOKENIZER}")'
    )
    conn.execute("INSERT INTO t(body) VALUES(?)", ("подписали договоры вчера",))

    for query in ("договор", "договором", "договоры"):
        count = conn.execute(
            "SELECT count(*) FROM t WHERE t MATCH ?", (query,)
        ).fetchone()[0]
        assert count == 1, query


def test_tokenize_spec_matches_end_to_end_pairing() -> None:
    """AC#1: the single-sourced ``TOKENIZE_SPEC`` is the pairing 3.2 will use.

    Building the ``tokenize=`` clause from ``TOKENIZE_SPEC`` (not literals) must
    yield a working index — proving the single source is the real contract.
    """
    conn = _conn()
    ru_lemma.register(conn)
    spec = " ".join(ru_lemma.TOKENIZE_SPEC)
    conn.execute(f'CREATE VIRTUAL TABLE t USING fts5(body, tokenize="{spec}")')
    conn.execute("INSERT INTO t(body) VALUES(?)", ("договоры",))
    count = conn.execute(
        "SELECT count(*) FROM t WHERE t MATCH ?", ("договор",)
    ).fetchone()[0]
    assert count == 1
    # The name half of the spec IS the single-sourced constant.
    assert ru_lemma.TOKENIZE_SPEC[0] == ru_lemma.TOKENIZER_NAME


def test_tokenizer_name_is_single_sourced() -> None:
    """AC#1: the fixed name is a real constant, re-exported from ``tgkb.common``.

    Mirrors ``test_throttle.py::test_default_rate_is_single_sourced`` — the value
    both processes register under must be ONE named constant, not a literal
    typed twice.
    """
    from tgkb import common

    assert ru_lemma.TOKENIZER_NAME == "ru_lemma"
    # The same object is re-exported from the package (imported, not reimplemented).
    assert common.TOKENIZER_NAME is ru_lemma.TOKENIZER_NAME
    assert common.register is ru_lemma.register
    assert common.lemmatize is ru_lemma.lemmatize


def test_analyzer_built_at_most_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards the "build the analyzer ONCE" requirement (a real perf trap).

    Many ``lemmatize`` calls must share a single ``MorphAnalyzer`` — a later
    refactor must not silently reintroduce per-token dict loads. We count real
    constructions by wrapping the constructor, having first cleared the cached
    singleton so the count starts from zero.
    """
    monkeypatch.setattr(ru_lemma, "_ANALYZER", None)

    calls = {"n": 0}
    real_ctor = ru_lemma.pymorphy3.MorphAnalyzer

    def counting_ctor(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real_ctor(*args, **kwargs)

    monkeypatch.setattr(ru_lemma.pymorphy3, "MorphAnalyzer", counting_ctor)

    for _ in range(5):
        ru_lemma.lemmatize("договоры")

    assert calls["n"] == 1
    # And the cached analyzer is the SAME object across calls.
    assert ru_lemma._analyzer() is ru_lemma._analyzer()
