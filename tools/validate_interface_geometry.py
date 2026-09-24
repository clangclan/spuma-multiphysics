#!/usr/bin/env python3
"""Build and run the bounded structured-interface CPU/CUDA validation."""
from pathlib import Path
import fcntl, hashlib, json, os, subprocess, tempfile, time

ROOT = Path(__file__).resolve().parents[1]
NVCC = Path("/home/jsw/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/bin/nvcc")
LOCK = Path("/home/jsw/cae-benchmark/run.lock")

def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    result = subprocess.run([str(x) for x in cmd], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE)
    print(result.stdout, end="")
    return result.stdout.strip()

def main():
    with tempfile.TemporaryDirectory(prefix="interface-geometry-") as td:
        td = Path(td)
        cpu = td / "cpu"
        gpu = td / "gpu"
        run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-pedantic", "tools/interface_geometry_test.cpp", "-o", cpu])
        cpu_result = run([cpu])
        run([NVCC, "-std=c++17", "-O2", "-arch=sm_120", "tools/interface_geometry_cuda_test.cu", "-o", gpu])
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        with LOCK.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            gpu_result = run([gpu])
        sources = [ROOT / "src/reactiveInterface/reactiveInterfaceGeometry.h", ROOT / "tools/interface_geometry_test.cpp", ROOT / "tools/interface_geometry_cuda_test.cu", Path(__file__).resolve()]
        evidence = {"schema": "reactive-interface-geometry-v1", "generatedUnix": time.time(), "scope": "standalone structured-grid module; timings are not end-to-end solver timings", "cpu": cpu_result, "cuda": gpu_result, "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
        out = ROOT / "results/spray-physics/interface-geometry.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, out)
        print(out.relative_to(ROOT))

if __name__ == "__main__":
    main()
