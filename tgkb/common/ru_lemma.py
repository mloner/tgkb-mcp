"""The single shared Russian lemmatizing FTS5 tokenizer (Story 3.1).

This module is the ONE code object that turns Russian text into FTS5 tokens by
lemmatizing every Cyrillic word to *all* its ranked pymorphy3 normal-forms. It
is registered under one fixed name (``ru_lemma``) and is imported — never
reimplemented — by both the ``writer`` and the ``reader`` (Story 3.2 wires it
into ``messages_fts`` on every writer/reader connection). That shared-object
shape is what makes index-side and query-side lemmatization identical *by
construction*.

Architecture invariants enforced here:

- **AD-6 / AD-11**: Russian morphology is a *single* custom FTS5 tokenizer,
  registered under the *same name with the same logic* on every connection in
  both processes. There is exactly ONE object, in ``common/``, under ONE fixed
  name (:data:`TOKENIZER_NAME`), with ONE homonymy policy (emit all ranked
  lemma hypotheses). "Same name, same logic" as prose is unverifiable; making
  it one imported code object makes it true structurally — nothing can diverge
  in-process. The *runtime* drift guard (a tokenizer fingerprint) is Story 3.3;
  3.1 makes drift structurally impossible in-process by there being nothing to
  diverge.
- **AD-6 (homonymy / recall policy, SM-2)**: a word with several morphological
  hypotheses (a homonym) emits ALL its distinct ranked lemmas — recall favored
  over precision. Both ends behave identically because the policy lives in this
  one shared object.
- **AD-7 (deterministic core)**: pymorphy3 is rule-based morphology — no model,
  no API key, no clock, no randomness, no network. Same input → same output.
- **AD-1**: this module lives under ``common/`` and imports ONLY stdlib +
  ``apsw`` / ``apsw.fts5`` + ``pymorphy3``. It is a pure text→tokens *leaf*: it
  MUST NEVER import ``tgkb.writer`` / ``tgkb.reader``, and it needs no sibling
  ``common`` module either (no DB, no config).

**Built here (Story 3.3 — the fingerprint half of AD-11):**

- The tokenizer **fingerprint** (:func:`fingerprint`, composed of the pinned
  ``pymorphy3-dicts-ru`` dict version + the ``pymorphy3`` engine version + the
  hand-maintained :data:`HYPOTHESIS_POLICY` tag). It lives HERE, with the
  tokenizer it fingerprints, so the fingerprint and the logic it fingerprints can
  never diverge — exactly like :data:`TOKENIZE_SPEC` is single-sourced with the
  ``register`` it feeds. Persistence of that fingerprint in the DB, the startup
  drift-check, and the forced ``'rebuild'`` on mismatch are Story 3.3's job too,
  but they live in :mod:`tgkb.common.schema` (a DB / FTS-lifecycle concern); this
  module only *computes* the fingerprint from its own pinned inputs.

**Not built here (deliberate seams):**

- ``messages_fts`` / any ``CREATE VIRTUAL TABLE`` in the real schema — that is
  **Story 3.2** (external-content FTS5 owned by ``common/``, maintained in the
  write transaction, AD-12). It will declare ``tokenize="ru_lemma unicode61"``
  (see :data:`TOKENIZE_SPEC`) and call :func:`register` on every writer/reader
  connection. The in-memory FTS5 table in the tests is a *behavioral proof of
  this tokenizer*, not the real index.
- **Per-connection registration on real DB connections** — no change to
  ``db.open_writer`` / ``db.open_reader``. Story 3.2 owns *when* to call
  :func:`register` (at the FTS lifecycle), so a plain DB open never pays the
  dictionary-load cost.
"""

from __future__ import annotations

import importlib.metadata
import threading
from typing import List, Sequence, Tuple

import apsw
import apsw.fts5
import pymorphy3

# --- The one fixed name — single-sourced (AD-6 / AD-11) ----------------------
# Both `writer/` and `reader/` register the tokenizer under THIS exact name in
# Story 3.2 (and every `CREATE VIRTUAL TABLE … tokenize="ru_lemma unicode61"`
# uses it). Never a string literal typed twice — mirrors db.DB_FILENAME /
# config.CONFIG_FILENAME.
TOKENIZER_NAME = "ru_lemma"

# The base tokenizer that does the actual Unicode word-segmentation. `unicode61`
# is SQLite's built-in Unicode-aware tokenizer — it segments Cyrillic correctly
# (AC#3 is satisfied by lemmatizing ON TOP OF unicode61, which unicode61 alone
# does not do). Single-sourced so Story 3.2 never re-types the pairing.
BASE_TOKENIZER = "unicode61"

# The full tokenize spec for the FTS5 virtual table, single-sourced so Story 3.2
# declares `tokenize="ru_lemma unicode61"` from ONE place (the "same logic both
# ends" contract, AD-11). The `ru_lemma` factory is base-agnostic — the caller
# (3.2's `CREATE VIRTUAL TABLE`) supplies `unicode61` as the base tokenizer arg
# — so this pairing is the single seam 3.2 reuses.
TOKENIZE_SPEC: Tuple[str, str] = (TOKENIZER_NAME, BASE_TOKENIZER)

# --- The fingerprint (Story 3.3, AD-11) -------------------------------------
# The distributions whose installed versions decide what lemmas the tokenizer
# emits. Read at RUNTIME (never hardcode the pinned literal) so a real
# `uv sync`-driven bump is actually DETECTED — hardcoding the frozen version
# would make a bumped dict still "match" the literal and defeat the whole guard.
#   * pymorphy3-dicts-ru — the dictionary; a bump changes which normal-forms
#     pymorphy3 emits → recall changes → the on-disk index must be rebuilt.
#   * pymorphy3 — the engine. AD-11 names only "dict version", but the INTENT is
#     "anything that changes lemma output"; a pymorphy3 engine change can alter
#     parses (and thus lemmas) with no dict bump, so it is included too.
_DICT_DISTRIBUTION = "pymorphy3-dicts-ru"
_ENGINE_DISTRIBUTION = "pymorphy3"

# The hypothesis-policy tag — the ONE HAND-MAINTAINED part of the guard.
#
# It names the CURRENT homonymy policy of `lemmatize()` below ("emit ALL distinct
# ranked normal-forms, first-seen-wins de-dup, surface-form fallback"). A policy
# change (e.g. switching to top-1-lemma-only) collapses recall WITHOUT any dict
# or engine version change — the version components alone cannot catch it, only
# this tag can. THEREFORE: **whenever `lemmatize()`'s semantics change, BUMP this
# tag** (e.g. `-v2`). That bump is the sanctioned, required action — it forces a
# drift-guarded rebuild on the next start so the on-disk index is re-derived
# under the new policy. This is the single non-automatic obligation in AD-11.
HYPOTHESIS_POLICY = "all-ranked-normal-forms-v1"

# The fingerprint is composed with `;` and `=` separators (see `fingerprint()`).
# A policy tag containing either would make the composed string ambiguous and
# could let two genuinely different tokenizer states collide (defeating drift
# detection). Enforce a safe charset at import time so a careless future bump
# fails LOUDLY here rather than silently weakening the guard.
assert (
    HYPOTHESIS_POLICY
    and ";" not in HYPOTHESIS_POLICY
    and "=" not in HYPOTHESIS_POLICY
), "HYPOTHESIS_POLICY must be non-empty and contain no ';' or '=' (fingerprint separators)"


def fingerprint() -> str:
    """Return the tokenizer's drift fingerprint (Story 3.3, AC#1 / AD-11).

    A single deterministic string composed of the three inputs that decide what
    lemmas this tokenizer emits — the pinned dict version, the pymorphy3 engine
    version, and the hand-maintained :data:`HYPOTHESIS_POLICY` tag::

        dicts=2.4.417150.4580142;engine=2.0.6;policy=all-ranked-normal-forms-v1

    Persisted in the DB when the index is built (:mod:`tgkb.common.schema`) and
    compared at startup by BOTH processes; a mismatch forces a mandatory FTS
    rebuild before serving. The composition lives in THIS one function so the
    persist-side and the check-side always compute it identically — if they ever
    disagreed, a matching index would look stale (needless rebuild) or a stale
    one would look current (the silent recall collapse AD-11 exists to prevent).

    **Determinism (AD-7):** a pure function of the installed distribution versions
    plus the constant policy tag — no clock, no randomness, no network, no AI.
    Same installed versions + same tag → same string, always.

    The versions are READ at runtime via :mod:`importlib.metadata` (stdlib, no new
    dependency) — never the frozen literal — so a genuine dict/engine bump is
    actually detected rather than masked by a hardcoded value.

    **Fail-safe on missing metadata:** if a distribution's version cannot be
    resolved (:class:`importlib.metadata.PackageNotFoundError` — e.g. a vendored /
    frozen install where ``.dist-info`` is absent), the component degrades to the
    ``unknown`` sentinel rather than raising. ``fingerprint()`` sits on the
    MANDATORY writer-open and reader-startup path, so a raise here would hard-fail
    ``export`` and the reader server, not just the guard. Degrading means the guard
    stays conservative: an ``unknown`` component will not match a real stored
    fingerprint, so the writer rebuilds and the reader refuses — fail-closed, never
    a silent stale-index serve.
    """
    dict_version = _distribution_version(_DICT_DISTRIBUTION)
    engine_version = _distribution_version(_ENGINE_DISTRIBUTION)
    return (
        f"dicts={dict_version};"
        f"engine={engine_version};"
        f"policy={HYPOTHESIS_POLICY}"
    )


def _distribution_version(distribution: str) -> str:
    """Installed version of ``distribution``, or ``"unknown"`` if unresolvable.

    See :func:`fingerprint` — a ``PackageNotFoundError`` must not crash the
    mandatory open path; the ``unknown`` sentinel keeps the guard fail-closed.
    """
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# Module-level lazy singleton for the expensive analyzer (see `_analyzer`).
_ANALYZER: "pymorphy3.MorphAnalyzer | None" = None
# Guards the lazy construction of `_ANALYZER` so a concurrent first call (e.g.
# two reader threads under the MCP server) cannot each build the multi-MB
# analyzer — see the thread-safety note in `_analyzer`.
_ANALYZER_LOCK = threading.Lock()


def _analyzer() -> pymorphy3.MorphAnalyzer:
    """Return the process-wide ``MorphAnalyzer``, constructing it once.

    ``pymorphy3.MorphAnalyzer()`` loads the whole ``pymorphy3-dicts-ru``
    dictionary from disk — hundreds of milliseconds, tens of MB. The tokenizer
    callback runs once PER TOKEN over potentially hundreds of thousands of
    messages (Story 3.2 indexing), so constructing an analyzer per token would
    make indexing pathologically slow. Build it ONCE and reuse.

    The singleton is *lazy* (not an import-time global) so that merely importing
    ``tgkb.common`` — which the AD-1 scanner and ``common/__init__.py`` do — does
    not pay the dict-load cost until search/index actually runs.

    Thread-safety: construction is guarded by :data:`_ANALYZER_LOCK` with a
    double-checked pattern, so a concurrent first call from multiple threads (the
    reader may serve queries on more than one thread) builds the analyzer exactly
    once rather than racing to load the dictionary twice. Once built,
    ``MorphAnalyzer.parse`` is read-only over an immutable loaded dictionary, so
    it is safe to call lock-free from the FTS5 tokenizer callback on both the
    writer's ``BEGIN IMMEDIATE`` connection and the reader's short read
    transactions — both only *read* the analyzer.
    """
    analyzer = _ANALYZER
    if analyzer is None:
        with _ANALYZER_LOCK:
            # Re-check under the lock: another thread may have built it while we
            # waited for the lock (double-checked locking).
            analyzer = _ANALYZER
            if analyzer is None:
                analyzer = pymorphy3.MorphAnalyzer()
                globals()["_ANALYZER"] = analyzer
    return analyzer


def lemmatize(token: str) -> List[str]:
    """Return every distinct pymorphy3 normal-form for ``token``, rank-ordered.

    This IS the SM-2 recall-favored homonymy policy (AC#2), living in the one
    shared object so both ends behave identically:

    - ``_analyzer().parse(token)`` returns ``Parse`` objects best-score-first.
    - We take ``p.normal_form`` from EVERY parse and de-duplicate **preserving
      pymorphy3's rank order** (first-seen wins). pymorphy3 can emit the same
      ``normal_form`` more than once — e.g. ``parse("договоры")`` yields
      ``договор`` twice → ``["договор"]``; ``parse("стали")`` yields ``стать``
      then several ``сталь`` → ``["стать", "сталь"]`` (a genuine homonym where
      BOTH lemmas are kept: a query for either should find this token).

    Emitting all distinct hypotheses is what surfaces, downstream, as multiple
    co-located tokens at one FTS5 position, so ``MATCH`` on any of them hits.

    **Empty/unparseable guard:** if no usable lemma comes back (rare — pymorphy3
    normally returns the surface form as its own normal_form even for unknown
    words), fall back to ``[token]``, the surface form. We deliberately never
    return ``[]``: ``TransformTokenizer`` treats an empty list as "remove the
    token", and silently dropping an un-lemmatizable Cyrillic token would be a
    latent recall hole (the token would be unfindable). The surface form keeps
    it at least literally matchable. An empty-string ``normal_form`` (which the
    base ``unicode61`` tokenizer never emits, but a direct caller could hit with
    ``""``) is skipped rather than indexed as a blank term; if that leaves no
    lemma AND the surface token is itself empty, we emit nothing (there is no
    meaningful token to keep).
    """
    lemmas: List[str] = []
    for parse in _analyzer().parse(token):
        normal_form = parse.normal_form
        # Skip a blank normal_form: indexing "" is a garbage searchable term.
        # `is not None` would still admit "", so test for a truthy string.
        if normal_form and normal_form not in lemmas:
            lemmas.append(normal_form)
    if lemmas:
        return lemmas
    # Fallback: keep the surface form so the token stays findable — but never
    # emit a blank token (an empty surface form is nothing worth indexing).
    return [token] if token else []


def register(conn: apsw.Connection) -> None:
    """Register the ``ru_lemma`` tokenizer on ``conn`` (AC#1).

    Wraps the pure :func:`lemmatize` transform in
    :class:`apsw.fts5.TransformTokenizer` and registers the resulting factory
    under :data:`TOKENIZER_NAME`. The factory is *base-agnostic*: the base
    tokenizer that does Unicode word-segmentation is supplied by the caller as a
    tokenizer arg (``tokenize="ru_lemma unicode61"`` in Story 3.2's
    ``CREATE VIRTUAL TABLE``; see :data:`TOKENIZE_SPEC`). The transform then
    rewrites each ``unicode61``-segmented token into its lemma hypotheses.

    Registration is per-connection: Story 3.2 calls this on EVERY writer and
    reader connection that will index or query ``messages_fts``. This module
    deliberately does NOT wire it into ``db.open_writer`` / ``db.open_reader``.
    """
    conn.register_fts5_tokenizer(
        TOKENIZER_NAME, apsw.fts5.TransformTokenizer(lemmatize)
    )


def tokenize(conn: apsw.Connection, text: str) -> List[Tuple[str, ...]]:
    """Tokenize ``text`` through ``ru_lemma`` WITHOUT a full FTS5 table.

    A thin testability + Story-3.2 seam: it registers ``ru_lemma`` on ``conn``,
    obtains the tokenizer bound to the :data:`BASE_TOKENIZER` base, and invokes
    it directly, returning one tuple of co-located tokens per source position.

    The APSW 3.53 call signature is exact:
    ``tok(utf8, apsw.FTS5_TOKENIZE_DOCUMENT, locale, *, include_offsets,
    include_colocated)`` — the third positional ``locale`` is REQUIRED (pass
    ``None``). With ``include_colocated=True`` (default) each position yields a
    tuple of co-located tokens — this is how a homonym's multiple lemmas share
    one FTS5 position (the mechanism AC#2 requires). ``include_offsets=False``
    drops the byte offsets, leaving clean token tuples for assertions.

    This is not a public API — its purpose is direct behavioral assertion of the
    tokenizer and giving Story 3.2 a proven invocation recipe.
    """
    register(conn)
    tok = conn.fts5_tokenizer(TOKENIZER_NAME, [BASE_TOKENIZER])
    result: Sequence[Tuple[str, ...]] = tok(
        text.encode("utf-8"),
        apsw.FTS5_TOKENIZE_DOCUMENT,
        None,
        include_offsets=False,
        include_colocated=True,
    )
    return [tuple(position) for position in result]
