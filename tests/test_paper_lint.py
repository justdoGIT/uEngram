from scripts.lint_paper import lint_markdown


PASSING_PAPER = """# A paper: short title

A statement citing [1] and [1–2]. Figure 1 illustrates the result. Table 1 reports it. Algorithm 1 describes it.

![Figure 1. Example](figures/example.svg)

**Table 1.** Example

| Measure | Value |
| --- | --- |
| A | 1 |

```algorithm | Algorithm 1. Example
return value
```

## References

1. Reference one, with artifact `results/example.json`.
2. Reference two.

## Data and Code Availability

Details are in `scripts/private.py`, repository version v8.
"""


def test_accepts_clean_standalone_paper_and_excluded_sections():
    assert lint_markdown(PASSING_PAPER) == []


def test_reports_each_repository_leakage_rule():
    paper = """# Title

The implementation lives in `snake_case`, scripts/main.py, at line:12-34.
The repository contains TODO, v2_internal, v8 and [INFERENCE], plus untracked files and commit abcdef0123456789.

## References

1. Citation.
"""
    messages = [message for _, message in lint_markdown(paper)]
    assert any("repository paths" in message for message in messages)
    assert any("file extensions" in message for message in messages)
    assert any("line-range citations" in message for message in messages)
    assert any("commit hashes" in message for message in messages)
    assert any("internal labels" in message for message in messages)
    assert any("repository terminology" in message for message in messages)
    assert any("backticked Python identifiers" in message for message in messages)


def test_checks_missing_and_uncited_references():
    paper = """# Title

An unsupported source [2].

## References

1. Never cited.
"""
    messages = [message for _, message in lint_markdown(paper)]
    assert "citation [2] has no matching reference" in messages
    assert "reference 1 is never cited" in messages


def test_checks_consecutive_and_text_referenced_visuals_and_algorithms():
    paper = """# Title

The table is useful.

![Figure 2. Missing prior figure](figure.svg)

**Table 2.** Unreferenced table

| A | B |
| --- | --- |
| 1 | 2 |

```algorithm | Algorithm 2. Unreferenced algorithm
return value
```
"""
    messages = [message for _, message in lint_markdown(paper)]
    assert any("Figure numbering must be consecutive" in message for message in messages)
    assert any("Figure 2 is not referenced" in message for message in messages)
    assert any("Table numbering must be consecutive" in message for message in messages)
    assert any("Table 2 is not referenced" in message for message in messages)
    assert any("Algorithm numbering must be consecutive" in message for message in messages)
    assert any("Algorithm 2 is not referenced" in message for message in messages)
