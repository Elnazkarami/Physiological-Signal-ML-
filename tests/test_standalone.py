"""The README's standalone claim, asserted rather than described.

The README says this library runs on its own: that the provenance core needs
no third-party package, and that the integration with the external Clinical
Data Fabric System is confined to one subpackage nobody has to install. Both
are the kind of claim that is true when written and quietly false three
commits later, when something convenient gets imported at the top of a module.

So they are tests. No numpy here, deliberately -- these run in the CI job that
installs no scientific stack, which is the only environment where the first
claim can be checked at all.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "physioml"

#: The one subpackage allowed to know CDFS exists.
INTEGRATION = SRC / "cdfs"


def imports_of(path: Path) -> set[str]:
    """Top-level module names this file imports, however it imports them."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def modules() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_the_package_declares_no_runtime_dependency():
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert manifest["project"]["dependencies"] == [], (
        "a runtime dependency would make `pip install physioml` pull a "
        "scientific stack in, which the README says it does not"
    )


def test_only_the_integration_subpackage_imports_cdfs():
    """Nothing in the pipeline reaches for the external provenance engine.

    If this fails, the library no longer runs without CDFS and the README is
    wrong -- regardless of whether anything visibly breaks, because the import
    would only fail on a machine that does not have it.
    """
    offenders = {
        str(path.relative_to(ROOT)): sorted(i for i in imports_of(path) if i == "cdfs")
        for path in modules()
        if INTEGRATION not in path.parents and "cdfs" in imports_of(path)
    }
    assert not offenders, f"cdfs imported outside physioml/cdfs/: {offenders}"


def test_the_provenance_core_imports_nothing_third_party():
    """``physioml.core`` is the part that must work in a bare interpreter."""
    allowed = {"physioml", "__future__"}
    stdlib = set(__import__("sys").stdlib_module_names)
    offenders: dict[str, list[str]] = {}
    for path in sorted((SRC / "core").rglob("*.py")):
        outside = sorted(imports_of(path) - allowed - stdlib)
        if outside:
            offenders[str(path.relative_to(ROOT))] = outside
    assert not offenders, f"the core gained a third-party import: {offenders}"


def test_exporting_a_prediction_needs_no_external_service():
    """The traceability export is the feature; it must not need a deployment.

    A prediction that can only be made traceable by writing it to CDFS would
    make the headline capability conditional on a private system, which is the
    opposite of what the README claims and of what ``examples/quickstart.py``
    demonstrates.
    """
    outside = imports_of(SRC / "evaluation" / "export.py")
    stdlib = set(__import__("sys").stdlib_module_names)
    third_party = sorted(outside - {"physioml", "__future__"} - stdlib)
    assert third_party == ["numpy"], (
        f"the export reaches outside physioml and numpy: {third_party}"
    )


def test_the_example_is_runnable_without_the_dataset_extras_it_does_not_name():
    """``examples/quickstart.py`` must only need what the README tells people
    to install."""
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = manifest["project"]["optional-dependencies"]
    documented = {
        name.split("[")[0].split(">")[0].split("=")[0].strip()
        for extra in ("signal", "ml")
        for name in extras[extra]
    }
    stdlib = set(__import__("sys").stdlib_module_names)
    needed = imports_of(ROOT / "examples" / "quickstart.py") - stdlib - {"physioml"}
    # numpy arrives with either extra; nothing else may be required.
    assert needed <= documented | {"numpy"}, (
        f"the example imports {sorted(needed - documented - {'numpy'})}, which "
        'pip install -e ".[signal,ml]" does not provide'
    )
