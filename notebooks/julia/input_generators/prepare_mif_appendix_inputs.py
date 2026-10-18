"""Create local public-package structure inputs for the Julia appendix notebooks."""

import ast
import csv
import io
import json
import tomllib
from pathlib import Path

from input_common import PYTHON, ROOT


SOURCE_CHECKS = {
    "root": ("__init__.py", "def __getattr__"),
    "registry": ("core/registry.py", "class Registry"),
    "base": ("core/base.py", "class Estimator(Protocol)"),
    "strategy": ("features/feature_importance/feature_importance_strategy.py", "class FeatureImportanceStrategy"),
    "bars": ("data/structures/abstract_bars.py", "class AbstractBars"),
    "deprecation": ("_deprecation.py", "def deprecated"),
}


def main() -> None:
    root_source = (PYTHON / "__init__.py").read_text(encoding="utf-8")
    root_tree = ast.parse(root_source)
    assignments = {
        target.id: node
        for node in root_tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    version = ast.literal_eval(assignments["__version__"].value)
    public_names = ast.literal_eval(assignments["__all__"].value)
    if version != "3.2.0":
        raise ValueError("RiskLabAI 3.2.0 is required")
    modules = sorted(path for path in PYTHON.rglob("*.py") if "__pycache__" not in path.parts)
    output = ROOT / "notebooks" / "julia" / "data" / "appendix_public_sources"
    output.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["Public source module"])
    writer.writerows((path.relative_to(PYTHON).as_posix(),) for path in modules)
    inventory = output / "python_module_inventory.csv"
    inventory.write_bytes(stream.getvalue().encode("utf-8"))
    declarations = {}
    for key, (relative, expected) in SOURCE_CHECKS.items():
        path = PYTHON / relative
        lines = path.read_text(encoding="utf-8").splitlines()
        line_number, declaration = next((index, line) for index, line in enumerate(lines, 1) if expected in line)
        declarations[key] = {"path": relative, "line_number": line_number, "declaration": declaration}
    area_node = assignments["__all__"]
    rows = [
        "python_version = " + json.dumps(version),
        "public_names = " + json.dumps(public_names),
        "public_names_declaration = " + json.dumps(ast.get_source_segment(root_source, area_node)),
    ]
    for key, record in declarations.items():
        rows.append("[declarations." + key + "]")
        rows.extend(name + " = " + (str(value) if isinstance(value, int) else json.dumps(value)) for name, value in record.items())
    manifest = output / "inputs.toml"
    manifest.write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
    tomllib.loads(manifest.read_text(encoding="utf-8"))
    print(json.dumps({"source_modules": len(modules), "declarations": len(declarations)}, indent=2))


if __name__ == "__main__":
    main()
