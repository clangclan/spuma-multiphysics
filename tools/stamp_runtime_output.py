#!/usr/bin/env python3
"""Forward child output unchanged and timestamp observed solver profile lines.

This does not create a new process group: the campaign owns and can terminate
the complete profiler/wrapper/time/solver process group. Observations are not
exact solver event times; buffering and scheduling can delay delivery.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A command is required')
    with args.output.open('x') as events:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            for line in proc.stdout:
                if line.startswith(b'REACTIVE_'):
                    events.write(json.dumps(dict(unixNs=time.time_ns(), monotonicNs=time.monotonic_ns(),
                        line=line.decode(errors='replace').rstrip('\n')), separators=(',', ':'))+'\n')
                    events.flush()
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
            code = proc.wait()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    return code if code >= 0 else 128-code


if __name__ == '__main__':
    raise SystemExit(main())
