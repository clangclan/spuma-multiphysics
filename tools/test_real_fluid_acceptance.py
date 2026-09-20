#!/usr/bin/env python3
"""The acceptance runner must not pass a successful child with missing checks."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_real_fluid_v21_local as runner

CHECKS=['V1','V2-V4','V3','V5-V6','legacy-gas','V6-recompute','internal-contract']

class AcceptanceEvidence(unittest.TestCase):
    def check_report(self, checks, expected):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);library=root/'fixture.so';library.write_bytes(b'test fixture')
            def execute(command,cwd,log,timeout):
                if checks is not None:
                    output=Path(command[command.index('--output')+1])
                    output.write_text(json.dumps({'tests':checks}))
                return {'status':'PASSED','returncode':0}
            argv=['runner','--thermo-dir',tmp,'--thermo-library',str(library),
                  '--transport-library',str(library),'--output-dir',str(root/'out')]
            with patch.object(sys,'argv',argv),patch.object(runner,'execute',execute),\
                 patch.object(runner.shutil,'which',return_value=None),contextlib.redirect_stdout(io.StringIO()):
                result=runner.main()
            report=json.loads((root/'out/acceptance.json').read_text())
            self.assertEqual(result,0 if expected else 1)
            self.assertEqual(report['all_requested_gates_passed'],expected)
    def test_missing_output(self):self.check_report(None,False)
    def test_missing_group(self):self.check_report([{'name':n,'status':'PASSED'} for n in CHECKS[:-1]],False)
    def test_skipped_group(self):self.check_report([{'name':n,'status':'SKIPPED' if n=='V3' else 'PASSED'} for n in CHECKS],False)
    def test_complete(self):self.check_report([{'name':n,'status':'PASSED'} for n in CHECKS],True)

if __name__=='__main__':unittest.main()
