#!/usr/bin/env python3
"""Check pruning policy transitions and reject unsupported host/device models.

Numerical parity is checked separately by replay_hem_precision.py using
--reference and --require-bitwise on captures from the intended campaign.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import re
import struct
import tempfile

from real_fluid_backend import RealFluidBackend
from validate_impinging_n2o_recovery import export_model, gpu_bind

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--library', type=Path, required=True)
    ap.add_argument('--cuda-library', type=Path, required=True)
    args = ap.parse_args()
    gpu = gpu_bind(args.cuda_library)
    checks = []

    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)

    def device_accepts(image, size):
        error = C.create_string_buffer(4096)
        handle = gpu.reactive_gpu_hem_create_v1(image, size, 8, error, len(error))
        if handle:
            gpu.reactive_gpu_hem_destroy_v1(handle)
        return bool(handle)

    liquid = ROOT/'examples/impinging-n2o-supercooled/cold-pr-148K-config.yaml'
    solid = ROOT/'examples/impinging-n2o-solid/cold-pr-148K-config.yaml'
    with tempfile.TemporaryDirectory() as temporary:
        # Model identity hashes resolve mechanism paths from the working
        # directory, so make these standalone test configurations explicit.
        configurations = []
        for name, source in [('liquid', liquid), ('solid', solid)]:
            target = Path(temporary)/(name+'.yaml')
            target.write_text(source.read_text().replace('mechanism: cold-pr-148K.yaml',
                'mechanism: '+str(source.parent/'cold-pr-148K.yaml')))
            configurations.append(target)
        liquid, solid = configurations
        gas = Path(temporary)/'gas.yaml'
        text = liquid.read_text()
        gas.write_text(re.sub(r'condensables:.*?(?=temperature-min:)', 'condensables: []\n', text, flags=re.S))
        for name, configuration, supported in [('liquid', liquid, True), ('solid', solid, False), ('gas-only', gas, False)]:
            with RealFluidBackend(configuration, args.library) as b:
                configure = b.lib.reactive_rt_set_gpu_hem_v1
                configure.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_char_p]
                configure.restype = C.c_int
                b.check(configure(b.handle, 1, 0, str(args.cuda_library.resolve()).encode()))
                select = b.lib.reactive_rt_set_gpu_hem_search_v1
                select.argtypes, select.restype = [C.c_void_p, C.c_int], C.c_int
                export = b.lib.reactive_rt_export_gpu_hem_v1
                export.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
                export.restype = C.c_int
                original = b.policy_hash
                image, size = export_model(b)
                check(name+'-reference-image', size == 47448 and struct.unpack_from('<i', image.raw, 45004)[0] == 0)
                check(name+'-reference-device', device_accepts(image, size))
                check(name+'-reject-invalid-flag', select(b.handle, 2) != 0 and b.policy_hash == original)
                status = select(b.handle, 1)
                check(name+'-host-pruning-scope', (status == 0) == supported)
                check(name+'-policy-transition', (b.policy_hash != original) == supported)
                if supported:
                    image, size = export_model(b)
                    check(name+'-exported-flag', struct.unpack_from('<i', image.raw, 45004)[0] == 1)
                    check(name+'-pruned-device', device_accepts(image, size))
                else:
                    # A raw capture can bypass the host setter. The device
                    # creation boundary must independently reject it.
                    modified = bytearray(image.raw[:size])
                    struct.pack_into('<i', modified, 45004, 1)
                    check(name+'-reject-raw-pruned-image', not device_accepts(C.create_string_buffer(bytes(modified)), size))
                check(name+'-restore-reference', select(b.handle, 0) == 0 and b.policy_hash == original)
    print(json.dumps(dict(passed=True, tests=checks), indent=2))


if __name__ == '__main__':
    main()
