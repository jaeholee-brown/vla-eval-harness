from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
COOKBOOK = REPO_ROOT / "docs" / "cookbook" / "adapter-authoring.md"
TEMPLATES = [
    REPO_ROOT / "vla_harness" / "adapters" / "policy" / "template_policy_adapter.py",
    REPO_ROOT / "vla_harness" / "adapters" / "embodiment" / "template_embodiment_adapter.py",
    REPO_ROOT / "vla_harness" / "eval" / "_skeleton.py",
]
REQUIRED_SENTENCE = (
    "If you can't fill this in from upstream docs alone, the protocol design has failed and we need to revisit it."
)


def test_cookbook_uses_numbered_sections_and_repeats_failure_clause():
    text = COOKBOOK.read_text(encoding="utf-8")
    numbered_sections = re.findall(r"^##?#+?\s*\d", text, flags=re.MULTILINE)
    assert numbered_sections, "Cookbook must use numbered sections."
    assert REQUIRED_SENTENCE in text
    assert text.count(REQUIRED_SENTENCE) >= 5


def test_every_template_todo_links_to_cookbook_section():
    pattern = re.compile(r"TODO\([^)]*cookbook §\d")
    for path in TEMPLATES:
        text = path.read_text(encoding="utf-8")
        todo_lines = [line for line in text.splitlines() if "TODO(" in line]
        assert todo_lines, f"{path.name} should contain TODO markers for adapter authors."
        for line in todo_lines:
            assert pattern.search(line), f"{path.name} TODO must reference a numbered cookbook section: {line!r}"
