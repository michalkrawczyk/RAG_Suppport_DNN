"""Generate a JSON map of modules -> classes/functions with their docstrings.

Scans specified directories (default: RAG_supporters/, tests/) using AST parsing
(no imports required, works without project dependencies installed).

Output format: list of objects, each representing one .py file.

Usage
-----
    python agents_notes/generate_module_map.py
    python agents_notes/generate_module_map.py --root /path/to/project
    python agents_notes/generate_module_map.py --dirs RAG_supporters tests examples
    python agents_notes/generate_module_map.py --output agents_notes/custom_map.json

Filtering examples (Python)
---------------------------
    import json
    data = json.load(open("agents_notes/module_map.json"))

    # All entries from the 'agents' submodule:
    [e for e in data if e["parent_module"] == "agents"]

    # A specific module by name:
    next(e for e in data if e["module"] == "domain_assesment")

    # Only production code (no tests):
    [e for e in data if e["package"] == "RAG_supporters"]

    # Modules containing a specific class:
    [e for e in data if "DomainAnalysisAgent" in e.get("classes", {})]

    # All files with a missing module docstring:
    [e for e in data if e["module_docstring"] is None and "error" not in e]
"""

import ast
import argparse
import json
import logging
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_SCAN_DIRS = ["RAG_supporters", "tests"]

# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _extract_methods(class_node: ast.ClassDef) -> dict[str, str | None]:
    """Extract method names and their docstrings from a class AST node.

    Parameters
    ----------
    class_node : ast.ClassDef
        The AST node representing the class.

    Returns
    -------
    dict[str, str | None]
        Mapping of method name -> docstring (None if absent).
    """
    methods: dict[str, str | None] = {}
    for node in ast.iter_child_nodes(class_node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods[node.name] = ast.get_docstring(node)
    return methods


# Node types at module level that may wrap class/function definitions
# (e.g. try/except for optional imports, if TYPE_CHECKING blocks, etc.)
_WRAPPER_NODES = (ast.Try, ast.TryStar, ast.If, ast.With, ast.AsyncWith)
# ast.TryStar is Python 3.11+; guard against older runtimes
_WRAPPER_TYPES = tuple(
    t for t in _WRAPPER_NODES if t is not None and isinstance(t, type)
)


def _iter_module_level_defs(
    parent: ast.AST,
) -> list[ast.stmt]:
    """Yield direct class/function definitions, descending into wrapper blocks.

    Wrapper blocks (Try, If, With) at module or wrapper level are entered
    recursively, so that definitions inside ``try/except ImportError`` guards
    or ``if TYPE_CHECKING:`` blocks are not missed.

    Parameters
    ----------
    parent : ast.AST
        A module node or a wrapper block node.

    Returns
    -------
    list[ast.stmt]
        Flat list of ClassDef / FunctionDef / AsyncFunctionDef nodes.
    """
    result: list[ast.stmt] = []
    children: list[ast.stmt] = list(ast.iter_child_nodes(parent))  # type: ignore[arg-type]
    for node in children:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            result.append(node)
        elif isinstance(node, _WRAPPER_TYPES):
            # Recurse into all child statement lists (body, handlers, orelse, finalbody)
            result.extend(_iter_module_level_defs(node))
    return result


def _extract_top_level_functions(
    module_node: ast.Module,
) -> dict[str, str | None]:
    """Extract top-level function names and their docstrings from a module AST.

    Descends into wrapper blocks (Try, If, With) to capture functions defined
    inside optional-import guards.

    Parameters
    ----------
    module_node : ast.Module
        The parsed AST of the module.

    Returns
    -------
    dict[str, str | None]
        Mapping of function name -> docstring (None if absent).
    """
    functions: dict[str, str | None] = {}
    for node in _iter_module_level_defs(module_node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = ast.get_docstring(node)
    return functions


def _extract_classes(module_node: ast.Module) -> dict[str, dict]:
    """Extract class names, their docstrings, and methods from a module AST.

    Descends into wrapper blocks (Try, If, With) to capture classes defined
    inside optional-import guards.

    Parameters
    ----------
    module_node : ast.Module
        The parsed AST of the module.

    Returns
    -------
    dict[str, dict]
        Mapping of class name -> {"docstring": ..., "methods": {...}}.
    """
    classes: dict[str, dict] = {}
    for node in _iter_module_level_defs(module_node):
        if isinstance(node, ast.ClassDef):
            classes[node.name] = {
                "docstring": ast.get_docstring(node),
                "methods": _extract_methods(node),
            }
    return classes


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------


def _build_metadata(rel_path: Path, root: Path) -> dict:
    """Build searchable metadata fields from a file's relative path.

    Parameters
    ----------
    rel_path : Path
        Path of the file relative to the project root.
    root : Path
        Absolute path of the project root (unused here, kept for symmetry).

    Returns
    -------
    dict
        Fields: path, file, module, parent_module, package.
    """
    parts = rel_path.parts  # e.g. ('RAG_supporters', 'agents', 'domain_assesment.py')
    return {
        "path": str(rel_path),
        "file": rel_path.name,
        "module": rel_path.stem,
        "parent_module": rel_path.parent.name,  # immediate parent dir
        "package": parts[0],  # top-level scan dir (RAG_supporters / tests)
    }


def process_file(abs_path: Path, rel_path: Path) -> dict:
    """Parse a single Python file and extract its structure.

    Parameters
    ----------
    abs_path : Path
        Absolute path to the Python file.
    rel_path : Path
        Path relative to the project root (used as the record identifier).

    Returns
    -------
    dict
        Record with metadata + module_docstring, classes, functions —
        or an error entry if the file cannot be parsed.
    """
    entry = _build_metadata(rel_path, abs_path.parent)

    try:
        source = abs_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(abs_path))
    except SyntaxError as exc:
        LOGGER.warning("SyntaxError in %s: %s", rel_path, exc)
        entry["error"] = f"SyntaxError: {exc}"
        return entry
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Cannot read %s: %s", rel_path, exc)
        entry["error"] = str(exc)
        return entry

    entry["module_docstring"] = ast.get_docstring(tree)
    entry["classes"] = _extract_classes(tree)
    entry["functions"] = _extract_top_level_functions(tree)
    return entry


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------


def scan_directories(root: Path, dirs: list[str]) -> list[dict]:
    """Scan the specified directories for Python files and extract their structure.

    Parameters
    ----------
    root : Path
        Absolute path of the project root.
    dirs : list[str]
        Directory names (relative to root) to scan.

    Returns
    -------
    list[dict]
        List of module records, one per .py file found.
    """
    records: list[dict] = []
    for dir_name in dirs:
        scan_root = root / dir_name
        if not scan_root.exists():
            LOGGER.warning("Directory not found, skipping: %s", scan_root)
            continue

        py_files = sorted(scan_root.rglob("*.py"))
        for abs_path in py_files:
            # Skip __pycache__ directories
            if "__pycache__" in abs_path.parts:
                continue
            rel_path = abs_path.relative_to(root)
            records.append(process_file(abs_path, rel_path))

    return records


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (defaults to sys.argv[1:]).

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    script_dir = Path(__file__).parent
    default_root = script_dir.parent
    default_output = script_dir / "module_map.json"

    parser = argparse.ArgumentParser(
        description="Generate a JSON map of modules -> classes/functions/docstrings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Project root directory (default: {default_root})",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=DEFAULT_SCAN_DIRS,
        metavar="DIR",
        help=f"Directories to scan, relative to --root (default: {DEFAULT_SCAN_DIRS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output JSON file path (default: {default_output})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the module map generation.

    Parameters
    ----------
    argv : list[str] | None
        Optional argument list for programmatic invocation.
    """
    args = parse_args(argv)
    root: Path = args.root.resolve()

    LOGGER.info("Project root : %s", root)
    LOGGER.info("Scanning     : %s", args.dirs)
    LOGGER.info("Output       : %s", args.output)

    records = scan_directories(root, args.dirs)

    errors = [r for r in records if "error" in r]
    ok = len(records) - len(errors)

    # Write output
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    LOGGER.info(
        "Done — processed %d file(s), %d error(s) → %s",
        ok,
        len(errors),
        output_path,
    )
    if errors:
        for e in errors:
            LOGGER.warning("  ! %s: %s", e["path"], e.get("error"))


if __name__ == "__main__":
    main()
