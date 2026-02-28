# Module Map — Usage Guide for Coding Agents

The module map is a pre-built JSON index of every class, function, method, docstring,
and type signature in the project.  
Use it to resolve symbol names, check signatures, and discover dependencies **without
reading source files**.

> **Note:** The generated `module_map.json` lives in `agent_ignore/` which is excluded
> from VS Code file explorer and Copilot search context.  The helper scripts in
> `agents_notes/` are still visible and usable.

---

## Quick Start

### 1. (Re-)generate the map

Run once, or after adding/changing modules:

```bash
python agents_notes/generate_module_map.py
# Output written to agent_ignore/module_map.json
```

Optional flags:

```bash
# Scan additional directories
python agents_notes/generate_module_map.py --dirs RAG_supporters tests examples

# Custom output path
python agents_notes/generate_module_map.py --output agent_ignore/custom_map.json

# Custom project root
python agents_notes/generate_module_map.py --root /path/to/project
```

### 2. Search the map

```bash
python agents_notes/search_module_map.py <QUERY> [OPTIONS]
```

---

## Search Examples

### Find a class

```bash
python agents_notes/search_module_map.py DatasetCheckAgent --type class
```

### Find a method across all classes

```bash
python agents_notes/search_module_map.py process_dataframe --type method
```

### Find a function by name fragment

```bash
python agents_notes/search_module_map.py rephrase --type function
```

### Find where a symbol is called (usage sites)

```bash
python agents_notes/search_module_map.py scan_directories --type usage
```

### Find methods/functions that accept a specific parameter

```bash
python agents_notes/search_module_map.py question_col --type param
```

### Narrow to a specific package or parent module

```bash
python agents_notes/search_module_map.py split --package RAG_supporters --parent dataset
```

### Output as JSON (for programmatic use)

```bash
python agents_notes/search_module_map.py DomainAnalysisAgent --type class --json
```

### Hide docstrings or method listings to reduce noise

```bash
python agents_notes/search_module_map.py augmentation --no-docstrings --no-methods
```

---

## Search Types Reference

| `--type` value | What is searched |
|----------------|-----------------|
| `all` (default) | Everything below |
| `module` | Module names and docstrings |
| `class` | Class names and docstrings |
| `method` | Method names inside classes |
| `function` | Module-level functions |
| `usage` | Call sites (`receiver.name(...)`) |
| `param` | Function/method parameter names |

Matching is **case-insensitive substring**.

---

## When to Use This Tool

| Scenario | Recommended action |
|----------|--------------------|
| Looking up a class signature before subclassing | `--type class` |
| Checking method parameters before calling | `--type method` or `--type param` |
| Finding all callers of a function | `--type usage` |
| Discovering which module exports a symbol | `--type all` |
| Cross-module refactoring impact analysis | `--type usage` + `--package` filter |

**Prefer this over opening source files** when you only need signatures or
want to confirm a symbol exists before making edits.

---

## Custom Map Path

If you generated the map to a non-default location:

```bash
python agents_notes/search_module_map.py MyClass --map /path/to/custom_map.json
```

---

## Output Format (JSON Records)

Each entry in `module_map.json` represents one `.py` file:

```json
{
  "path": "RAG_supporters/agents/domain_assesment.py",
  "module": "domain_assesment",
  "parent_module": "agents",
  "package": "RAG_supporters",
  "module_docstring": "...",
  "classes": {
    "DomainAnalysisAgent": {
      "docstring": "...",
      "bases": ["BaseAgent"],
      "line": 42,
      "methods": {
        "extract_domains_from_source": {
          "docstring": "...",
          "signature": { "params": [...] },
          "line": 88
        }
      }
    }
  },
  "functions": { ... },
  "calls": [ { "name": "...", "receiver": "...", "line": 12 } ],
  "constants": { "module_constants": ["LOGGER", "DEFAULT_SCAN_DIRS"], "__all__": null }
}
```
