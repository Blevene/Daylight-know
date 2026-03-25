# ELI5 Section — Design Spec

**Date:** 2026-03-25
**Issue:** [#6 — Feat: ELI5 Section](https://github.com/Blevene/Daylight-know/issues/6)

## Problem

The digest email's summary section is adequate but not always accessible to non-specialists. An "Explain Like I'm 5" section would re-explain each paper's key findings in plain, jargon-free language.

## Approach

Follow the existing post-processing pattern (implications, critiques) exactly: a new prompt file, a new function in `postprocessor.py`, a config toggle, and a new section in the email templates.

## Changes

### 1. New prompt — `src/digest_pipeline/prompts/eli5.md`

System prompt instructing the LLM to re-explain each paper in plain language. No jargon, no unexpanded acronyms, analogies encouraged. Output: a short paragraph (3-5 sentences) per paper.

### 2. New function — `postprocessor.py`

```python
ELI5_SYSTEM_PROMPT = load_prompt("eli5")

def generate_eli5(papers: list[Paper], settings: Settings) -> dict[str, str]:
```

Same contract as `extract_implications()` and `generate_critiques()`: calls `llm_call(papers, ELI5_SYSTEM_PROMPT, settings, label="eli5", schema_name="paper_eli5")`, applies `_normalize_markdown_bullets()` (harmless on paragraph text, maintains pattern consistency), returns `dict[str, str]` mapping `paper_N` keys to ELI5 text.

Note: Unlike implications/critiques which use structured bullet-point output, the ELI5 prompt intentionally produces flowing paragraphs. This is a deliberate divergence — do not add labeled sections to the prompt.

### 3. Config toggle — `config.py`

```python
postprocessing_eli5: bool = Field(default=True)
```

Added to the `# ── Post-processing (optional)` section alongside the existing toggles. Env var: `POSTPROCESSING_ELI5`.

### 4. Data model — `pipeline.py`

Add `eli5: str = ""` field to `PaperAnalysis` dataclass, after `summary` and before `implications`.

### 5. Pipeline wiring — `pipeline.py`

- Import `generate_eli5` from `postprocessor`
- Call `generate_eli5()` gated by `settings.postprocessing_eli5`, alongside implications/critiques
- Update `_build_analyses()` signature to accept `eli5: dict[str, str]` parameter
- Add missing-key warning for eli5 (matching the pattern on lines 78-81 for implications/critiques)
- Pass `eli5=eli5.get(key, "")` in the `PaperAnalysis` constructor
- Update the module docstring (line 8) to include ELI5 alongside implications & critiques

### 6. Email templates — `emailer.py`

Add ELI5 section immediately after Summary and before Practical Implications in both templates.

**HTML:**
```jinja2
{% if paper.eli5 %}
<h3>ELI5</h3>
<div class="section">{{ paper.eli5 | md }}</div>
{% endif %}
```

**Plain text:**
```jinja2
{% if paper.eli5 %}

### ELI5

{{ paper.eli5 }}
{% endif %}
```

### 7. `.env.example`

Add `POSTPROCESSING_ELI5=true` with a comment explaining the feature.

### 8. Setup wizard — `setup.py`

Add ELI5 toggle prompt following the pattern of existing post-processing toggles. Also add `"POSTPROCESSING_ELI5"` to the `"Post-processing"` group in `_write_env_file()`'s `sections` list.

### 9. Tests

Unit tests for `generate_eli5()` alongside existing postprocessor tests in `tests/unit/test_postprocessor.py`. Mirror all existing test cases: success path, max-tokens passthrough, empty content handling, malformed JSON, and schema key validation.

## Section Order in Email

1. Summary
2. **ELI5** ← new
3. Practical Implications
4. Critique

## Out of Scope

- No changes to the summarizer prompt itself
- No new LLM provider logic — reuses existing `llm_call()` infrastructure
- No changes to ranking, fetching, or storage
