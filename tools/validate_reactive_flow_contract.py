#!/usr/bin/env python3
"""Compile the source-derived Flow version mutation contract without SPUMA.

This checks the real member declaration, step signature and increment expressions
in isolation. It does NOT compile the full Flow body, SPUMA headers or CUDA path.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "src/reactiveFoam/ReactiveFoam.C")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite evidence")
    source = args.source.read_text()
    member = re.search(r"^\s*((?:mutable\s+)?uint64_t\s+transportVersion\s*=\s*0\s*;)", source, re.M)
    method = re.search(r"\b(void\s+step\s*\([^)]*\)\s*(?:const\s*)?)\{", source)
    if not member or not method:
        raise SystemExit("Cannot extract Flow mutation contract; update the extractor for the new source layout")
    start = method.end(); end = start; depth = 1
    while depth and end < len(source):
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    increments = re.findall(r"\+\+transportVersion\b", source[start:end])
    if depth or not increments:
        raise SystemExit("Cannot find complete step body and version mutations")
    unit = "\n".join([
        "#include <cstdint>", "#include <vector>",
        "using Array=std::vector<double>; using States=std::vector<int>;",
        "struct FlowMutationContract {", member[1], method[1]+"{",
        *[mutation+";" for mutation in increments], "}", "};", "",
    ])
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    with tempfile.TemporaryDirectory(prefix="reactive-flow-contract-") as directory:
        path = Path(directory) / "flow_contract.cpp"; path.write_text(unit)
        result = subprocess.run(compiler+["-std=c++17", "-fsyntax-only", str(path)], capture_output=True, text=True)
        diagnostic = result.stderr.replace(str(path), "flow_contract.cpp")
    non_const = not re.search(r"\)\s*const\s*$", method[1]) and "mutable" not in member[1]
    report = dict(passed=result.returncode == 0 and bool(non_const),
        scope="source-derived version mutation contract only; not a full SPUMA build",
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        member=member[1], signature=method[1].strip(), increments=len(increments),
        compiler=subprocess.check_output(compiler+["--version"], text=True).splitlines()[0],
        compiler_exit_code=result.returncode, diagnostic=diagnostic,
        non_const_step_and_non_mutable_version=bool(non_const), translation_unit=unit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
