from pathlib import Path

import scripts.build_results_appendix as builder


def test_print_mode_only_prints_rendered_ledger(monkeypatch, capsys):
    groups = [("test", "Test group", [{"outcome": "positive"}])]
    monkeypatch.setattr(builder, "collect", lambda repo: groups)
    monkeypatch.setattr(builder, "render", lambda actual, figure: "# ledger\n")
    monkeypatch.setattr(builder.sys, "argv", ["build_results_appendix.py", "--print"])

    builder.main()

    assert capsys.readouterr().out == "# ledger\n\n"


def test_write_mode_writes_ledger_and_figure_not_paper(tmp_path, monkeypatch):
    paper = tmp_path / "paper.md"
    paper.write_text("paper remains untouched", encoding="utf-8")
    ledger = tmp_path / "results-ledger.md"
    figure = tmp_path / "docs/research-paper/figures/results-ledger.svg"
    groups = [("test", "Test group", [{"outcome": "positive"}])]
    monkeypatch.setattr(builder, "collect", lambda repo: groups)
    monkeypatch.setattr(builder, "render", lambda actual, line: "## tables\n")
    monkeypatch.setattr(builder, "LEDGER", ledger)
    monkeypatch.setattr(builder, "REPO", tmp_path)
    monkeypatch.setattr(builder.sys, "argv", ["build_results_appendix.py"])

    import results_appendix.ledger_figure as ledger_figure
    def write_figure(actual, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("svg", encoding="utf-8")

    monkeypatch.setattr(ledger_figure, "write_ledger", write_figure)
    builder.main()

    assert ledger.read_text(encoding="utf-8") == "# Supplementary Results Ledger\n\n## tables\n"
    assert figure.read_text(encoding="utf-8") == "svg"
    assert paper.read_text(encoding="utf-8") == "paper remains untouched"
