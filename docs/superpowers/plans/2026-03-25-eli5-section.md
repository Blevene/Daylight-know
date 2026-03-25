# ELI5 Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ELI5 (Explain Like I'm 5) section to the digest email that re-explains each paper in plain, accessible language.

**Architecture:** Follows the existing post-processing pattern (implications, critiques) exactly — a new prompt, a new function in `postprocessor.py`, a config toggle, and new template sections. Each post-processor is a thin wrapper around the shared `llm_call()` infrastructure.

**Tech Stack:** Python, litellm, Jinja2, pytest

**Spec:** `docs/superpowers/specs/2026-03-25-eli5-section-design.md`

---

### Task 1: Add ELI5 prompt file

**Files:**
- Create: `src/digest_pipeline/prompts/eli5.md`

- [ ] **Step 1: Create the prompt file**

```markdown
You are a science communicator who makes complex research accessible to everyone. Given a set of academic paper abstracts, explain each paper's key findings in plain, everyday language that anyone could understand — no jargon, no unexpanded acronyms.

Use simple analogies where they help. Write a short paragraph (3-5 sentences) per paper. Imagine you are explaining this to a curious friend with no technical background.
```

- [ ] **Step 2: Verify the prompt loads**

Run: `python -c "from digest_pipeline.prompts import load_prompt; print(load_prompt('eli5')[:50])"`
Expected: First 50 chars of the prompt printed without error.

- [ ] **Step 3: Commit**

```bash
git add src/digest_pipeline/prompts/eli5.md
git commit -m "feat(eli5): add ELI5 system prompt"
```

---

### Task 2: Add `generate_eli5()` to postprocessor

**Files:**
- Modify: `src/digest_pipeline/postprocessor.py`
- Test: `tests/unit/test_postprocessor.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_postprocessor.py`:

```python
from digest_pipeline.postprocessor import (
    CRITIQUES_SYSTEM_PROMPT,
    ELI5_SYSTEM_PROMPT,
    IMPLICATIONS_SYSTEM_PROMPT,
    extract_implications,
    generate_critiques,
    generate_eli5,
)

# ... (update the existing import block, keep all existing tests)


@patch("digest_pipeline.llm_utils.litellm.completion")
def test_generate_eli5_success(mock_completion, make_paper, make_settings):
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({"paper_1": "Think of it like a recipe..."})
    mock_completion.return_value = MagicMock(choices=[mock_choice])

    result = generate_eli5([make_paper()], make_settings())

    assert result == {"paper_1": "Think of it like a recipe..."}
    mock_completion.assert_called_once()

    call_kwargs = mock_completion.call_args
    messages = call_kwargs.kwargs["messages"]
    assert messages[0]["content"] == ELI5_SYSTEM_PROMPT
    assert "response_format" in call_kwargs.kwargs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_postprocessor.py::test_generate_eli5_success -v`
Expected: FAIL with `ImportError` (generate_eli5 not defined yet)

- [ ] **Step 3: Write the implementation**

Update module docstring (line 1) from "practical implications and critiques" to "practical implications, critiques, and ELI5 explanations."

Add to `src/digest_pipeline/postprocessor.py`:

```python
ELI5_SYSTEM_PROMPT = load_prompt("eli5")


def generate_eli5(papers: list[Paper], settings: Settings) -> dict[str, str]:
    """Generate per-paper ELI5 (plain-language) explanations.

    Returns a dict mapping ``paper_N`` keys to ELI5 strings.
    Enforces ``settings.llm_max_tokens`` and retries with exponential
    backoff on rate-limit errors, matching the summarizer contract.
    """
    raw = llm_call(
        papers,
        ELI5_SYSTEM_PROMPT,
        settings,
        label="eli5",
        schema_name="paper_eli5",
    )
    return {k: _normalize_markdown_bullets(v) for k, v in raw.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_postprocessor.py::test_generate_eli5_success -v`
Expected: PASS

- [ ] **Step 5: Run all postprocessor tests**

Run: `pytest tests/unit/test_postprocessor.py -v`
Expected: All tests pass (existing + new)

- [ ] **Step 6: Commit**

```bash
git add src/digest_pipeline/postprocessor.py tests/unit/test_postprocessor.py
git commit -m "feat(eli5): add generate_eli5() to postprocessor"
```

---

### Task 3: Add config toggle

**Files:**
- Modify: `src/digest_pipeline/config.py:72-73`

- [ ] **Step 1: Add the config field**

Add after `postprocessing_critiques` (line 73):

```python
    postprocessing_eli5: bool = Field(default=True)
```

- [ ] **Step 2: Verify**

Run: `python -c "from digest_pipeline.config import Settings; s = Settings(_env_file=None, llm_api_key='x'); print(s.postprocessing_eli5)"`
Expected: `True`

- [ ] **Step 3: Commit**

```bash
git add src/digest_pipeline/config.py
git commit -m "feat(eli5): add postprocessing_eli5 config toggle"
```

---

### Task 4: Wire ELI5 into pipeline

**Files:**
- Modify: `src/digest_pipeline/pipeline.py:39,50-96,259-273`
- Test: `tests/unit/test_pipeline.py`

- [ ] **Step 1: Write the failing test for `_build_analyses` with ELI5**

Add to `tests/unit/test_pipeline.py`:

```python
def test_build_analyses_with_eli5():
    papers = [
        _make_paper(title="Paper 1", url="http://1", authors=["A"]),
    ]
    summaries = {"paper_1": "Sum 1"}
    implications = {"paper_1": "Impl 1"}
    critiques = {"paper_1": "Crit 1"}
    eli5 = {"paper_1": "Think of it like..."}

    analyses = _build_analyses(papers, summaries, implications, critiques, eli5=eli5)

    assert analyses[0].eli5 == "Think of it like..."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_pipeline.py::test_build_analyses_with_eli5 -v`
Expected: FAIL (PaperAnalysis has no `eli5` field, `_build_analyses` doesn't accept `eli5` kwarg)

- [ ] **Step 3: Update PaperAnalysis dataclass**

Add `eli5: str = ""` field after `summary` and before `implications` in the `PaperAnalysis` dataclass (line 61-63 area):

```python
@dataclass
class PaperAnalysis:
    """Per-paper grouped analysis for the digest email."""

    title: str
    url: str
    source: str = "arxiv"
    authors: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    upvotes: int = 0
    fields_of_study: list[str] = field(default_factory=list)
    summary: str = ""
    eli5: str = ""
    implications: str = ""
    critique: str = ""
```

- [ ] **Step 4: Update `_build_analyses()` signature and body**

Update the import line (line 39) to add `generate_eli5`:

```python
from digest_pipeline.postprocessor import extract_implications, generate_critiques, generate_eli5
```

Update `_build_analyses` to accept and wire `eli5`:

```python
def _build_analyses(
    papers: list[Paper],
    summaries: dict[str, str],
    implications: dict[str, str],
    critiques: dict[str, str],
    *,
    eli5: dict[str, str] | None = None,
) -> list[PaperAnalysis]:
    """Zip LLM results into per-paper PaperAnalysis objects."""
    if eli5 is None:
        eli5 = {}
    analyses: list[PaperAnalysis] = []
    for i, paper in enumerate(papers, 1):
        key = f"paper_{i}"
        if key not in summaries:
            logger.warning("Missing summary for %s (%s).", key, paper.title)
        if implications and key not in implications:
            logger.warning("Missing implications for %s (%s).", key, paper.title)
        if critiques and key not in critiques:
            logger.warning("Missing critique for %s (%s).", key, paper.title)
        if eli5 and key not in eli5:
            logger.warning("Missing ELI5 for %s (%s).", key, paper.title)
        analyses.append(
            PaperAnalysis(
                title=paper.title,
                url=paper.url,
                source=paper.source,
                authors=paper.authors,
                categories=paper.categories,
                upvotes=paper.upvotes,
                fields_of_study=paper.fields_of_study,
                summary=summaries.get(key, ""),
                eli5=eli5.get(key, ""),
                implications=implications.get(key, ""),
                critique=critiques.get(key, ""),
            )
        )
    return analyses
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_pipeline.py::test_build_analyses_with_eli5 -v`
Expected: PASS

- [ ] **Step 6: Write test for missing ELI5 key warning**

Add to `tests/unit/test_pipeline.py`:

```python
def test_build_analyses_warns_on_missing_eli5(caplog):
    papers = [
        _make_paper(title="Paper 1", url="http://1", authors=["A"]),
    ]
    summaries = {"paper_1": "Sum 1"}

    with caplog.at_level(logging.WARNING, logger="digest_pipeline.pipeline"):
        _build_analyses(papers, summaries, {}, {}, eli5={"paper_99": "stale"})

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Missing ELI5" in w and "paper_1" in w for w in warnings)
```

- [ ] **Step 7: Run test**

Run: `pytest tests/unit/test_pipeline.py::test_build_analyses_warns_on_missing_eli5 -v`
Expected: PASS

- [ ] **Step 8: Wire ELI5 into `run()` function**

Update the module docstring (line 8) to include ELI5:
```
  8. Post-process: ELI5, implications & critiques (per-paper JSON)
```

Add ELI5 call in the post-processing section (after line 266), and add error logging:

```python
    eli5_results: dict[str, str] = {}
    if settings.postprocessing_eli5:
        eli5_results = generate_eli5(processed_papers, settings)
        if not eli5_results:
            logger.error(
                "ELI5 generation returned no results for %d papers.", len(processed_papers)
            )
```

Update the `_build_analyses` call (line 273) to pass `eli5`:

```python
    analyses = _build_analyses(processed_papers, summaries, implications, critiques, eli5=eli5_results)
```

- [ ] **Step 9: Update ALL existing `run()` tests to mock `generate_eli5`**

After Step 8 wires `generate_eli5` into `run()`, every existing test that calls `run()` will fail unless `generate_eli5` is mocked. Add `@patch("digest_pipeline.pipeline.generate_eli5", return_value={})` to the decorator stack of ALL these tests (place between `generate_critiques` and `send_digest` patches), and add a `mock_eli5` parameter:

1. **`test_run_full_pipeline`** — use `return_value={"paper_1": "Simple explanation"}` instead of `{}`, add `mock_eli5` param, and assert:
   ```python
       mock_eli5.assert_called_once()
       assert analyses[0].eli5 == "Simple explanation"
   ```

2. **`test_run_postprocessing_disabled`** — add mock, add `mock_eli5` param, add `postprocessing_eli5=False` to settings, assert:
   ```python
       mock_eli5.assert_not_called()
       assert analyses[0].eli5 == ""
   ```

3. **`test_run_logs_error_when_critiques_empty`** — add mock, add `mock_eli5` param (no additional assertions needed, just prevent unpatched call).

4. **`test_pipeline_calls_ranker_for_openalex`** — add mock, add `mock_eli5` param, add `postprocessing_eli5=False` to settings.

5. **`test_arxiv_papers_ranked_when_interest_configured`** — add mock, add `mock_eli5` param (already has `postprocessing_eli5` defaulting, so add `postprocessing_eli5=False` to settings or add the mock with `return_value={}`).

**Important:** `@patch` decorators apply bottom-to-top. The `generate_eli5` patch should be placed in the decorator stack right above the `generate_critiques` patch (so the mock parameter appears right after `mock_critiques` in the function signature).

- [ ] **Step 10: Write test for ELI5 empty results error logging**

Add to `tests/unit/test_pipeline.py`:

```python
@patch("digest_pipeline.pipeline.save_seen")
@patch("digest_pipeline.pipeline.load_seen", return_value={})
@patch("digest_pipeline.pipeline.send_digest")
@patch("digest_pipeline.pipeline.generate_eli5", return_value={})
@patch("digest_pipeline.pipeline.generate_critiques", return_value={"paper_1": "Crit"})
@patch("digest_pipeline.pipeline.extract_implications", return_value={"paper_1": "Impl"})
@patch("digest_pipeline.pipeline.summarize", return_value={"paper_1": "Summary"})
@patch("digest_pipeline.pipeline.store_chunks", return_value=[])
@patch("digest_pipeline.pipeline.chunk_text", return_value=[])
@patch(
    "digest_pipeline.pipeline.extract_text",
    return_value=ExtractionResult(paper_id="2401.00001", text="content", parseable=True),
)
@patch("digest_pipeline.pipeline.fetch_papers")
def test_run_logs_error_when_eli5_empty(
    mock_fetch, mock_extract, mock_chunk, mock_store,
    mock_summarize, mock_implications, mock_critiques,
    mock_eli5, mock_email, mock_load_seen, mock_save_seen,
    caplog, make_settings,
):
    """Pipeline logs error when ELI5 generation returns empty."""
    mock_fetch.return_value = [_make_paper()]
    with caplog.at_level(logging.ERROR, logger="digest_pipeline.pipeline"):
        run(make_settings())

    assert any(
        "ELI5 generation returned no results" in r.message and r.levelno == logging.ERROR
        for r in caplog.records
    )
    mock_email.assert_called_once()
```

- [ ] **Step 11: Run all pipeline tests**

Run: `pytest tests/unit/test_pipeline.py -v`
Expected: All tests pass

- [ ] **Step 12: Commit**

```bash
git add src/digest_pipeline/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat(eli5): wire generate_eli5 into pipeline orchestrator"
```

---

### Task 5: Add ELI5 section to email templates

**Files:**
- Modify: `src/digest_pipeline/emailer.py:94-105,145-162`
- Test: `tests/unit/test_emailer.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_emailer.py`:

```python
def test_build_email_with_eli5(make_settings):
    settings = make_settings()
    papers = [
        PaperAnalysis(
            title="Test Paper",
            url="https://arxiv.org/abs/2401.00001",
            authors=["Alice"],
            summary="Test summary",
            eli5="Think of it like building with blocks.",
            implications="Apply this to X.",
            critique="Limitation in Y.",
        ),
    ]
    msg = _build_email(papers, "2025-01-15", settings)
    payloads = msg.get_payload()
    html_body = payloads[1].get_payload(decode=True).decode()
    plain_body = payloads[0].get_payload(decode=True).decode()
    assert "ELI5" in html_body
    assert "building with blocks" in html_body
    assert "ELI5" in plain_body
    assert "building with blocks" in plain_body


def test_build_email_without_eli5(make_settings):
    settings = make_settings()
    papers = [
        PaperAnalysis(
            title="Test Paper",
            url="https://arxiv.org/abs/2401.00001",
            authors=["Alice"],
            summary="Test summary",
        ),
    ]
    msg = _build_email(papers, "2025-01-15", settings)
    payloads = msg.get_payload()
    html_body = payloads[1].get_payload(decode=True).decode()
    assert ">ELI5<" not in html_body


def test_build_email_eli5_appears_between_summary_and_implications(make_settings):
    settings = make_settings()
    papers = [
        PaperAnalysis(
            title="Test Paper",
            url="https://arxiv.org/abs/2401.00001",
            authors=["Alice"],
            summary="Test summary",
            eli5="Simple explanation here.",
            implications="Apply this to X.",
        ),
    ]
    msg = _build_email(papers, "2025-01-15", settings)
    payloads = msg.get_payload()
    html_body = payloads[1].get_payload(decode=True).decode()
    summary_pos = html_body.index("Summary")
    eli5_pos = html_body.index("ELI5")
    implications_pos = html_body.index("Practical Implications")
    assert summary_pos < eli5_pos < implications_pos
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_emailer.py::test_build_email_with_eli5 -v`
Expected: FAIL (no ELI5 section in template)

- [ ] **Step 3: Add ELI5 to HTML template**

In `emailer.py`, after the Summary section (line 97) and before the Implications section (line 98), add:

```jinja2
    {% if paper.eli5 %}
    <h3>ELI5</h3>
    <div class="section">{{ paper.eli5 | md }}</div>
    {% endif %}
```

- [ ] **Step 4: Add ELI5 to plain text template**

In `emailer.py`, after the Summary section (line 150) and before the Implications section (line 151), add:

```jinja2
{% if paper.eli5 %}

### ELI5

{{ paper.eli5 }}
{% endif %}
```

- [ ] **Step 5: Run all emailer tests**

Run: `pytest tests/unit/test_emailer.py -v`
Expected: All tests pass (existing + new)

- [ ] **Step 6: Commit**

```bash
git add src/digest_pipeline/emailer.py tests/unit/test_emailer.py
git commit -m "feat(eli5): add ELI5 section to email templates"
```

---

### Task 6: Update .env.example, setup wizard, and config docs

**Files:**
- Modify: `.env.example:30-32`
- Modify: `src/digest_pipeline/setup.py:378-383,529`

- [ ] **Step 1: Update `.env.example`**

Add after `POSTPROCESSING_CRITIQUES="true"` (line 32):

```
POSTPROCESSING_ELI5="true"
```

Also update the section comment (line 30) from:
```
# ── Post-processing (practical implications & critiques) ──────
```
to:
```
# ── Post-processing (ELI5, implications & critiques) ─────────
```

- [ ] **Step 2: Update setup wizard toggle**

In `setup.py`, after the critiques toggle (line 383), add:

```python
    config["POSTPROCESSING_ELI5"] = (
        "true" if _prompt_bool("Enable ELI5 (plain-language) explanations?", default=True) else "false"
    )
```

- [ ] **Step 3: Update `_write_env_file` sections list**

In `setup.py`, update the Post-processing section (line 529) from:

```python
            ["POSTPROCESSING_IMPLICATIONS", "POSTPROCESSING_CRITIQUES"],
```

to:

```python
            ["POSTPROCESSING_IMPLICATIONS", "POSTPROCESSING_CRITIQUES", "POSTPROCESSING_ELI5"],
```

- [ ] **Step 4: Verify config loads with env var**

Run: `POSTPROCESSING_ELI5=false python -c "from digest_pipeline.config import Settings; s = Settings(_env_file=None, llm_api_key='x'); print(s.postprocessing_eli5)"`
Expected: `False`

- [ ] **Step 5: Commit**

```bash
git add .env.example src/digest_pipeline/setup.py
git commit -m "feat(eli5): add ELI5 to .env.example and setup wizard"
```

---

### Task 7: Final verification

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/unit/ -v`
Expected: All tests pass

- [ ] **Step 2: Run linting**

Run: `ruff check src/digest_pipeline/postprocessor.py src/digest_pipeline/pipeline.py src/digest_pipeline/emailer.py src/digest_pipeline/config.py`
Expected: No errors

- [ ] **Step 3: Dry-run the pipeline** (optional, requires API key)

Run: `digest-pipeline --dry-run`
Expected: ELI5 section appears in console output between Summary and Practical Implications
