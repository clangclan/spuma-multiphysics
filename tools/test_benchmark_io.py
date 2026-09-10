#!/usr/bin/env python3
"""Regression checks for the benchmark's actual field-file formats."""
import gzip
from pathlib import Path
import tempfile
import unittest
import numpy as np
import benchmark as b


class FieldIOTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_scalar_vector_formats(self):
        for vector in (False, True):
            expected = np.arange(9 if vector else 3).reshape(3, 3 if vector else 1)
            for fmt in ('ascii', 'binary'):
                for dtype in ('<f8', '>f4'):
                    endian = 'LSB' if dtype[0] == '<' else 'MSB'
                    header = (f'FoamFile {{ format {fmt}; class vol{"Vector" if vector else "Scalar"}Field; '
                              f'arch "{endian};label=32;scalar={64 if dtype[-1] == "8" else 32}"; }}\n'
                              f'internalField nonuniform List<{"vector" if vector else "scalar"}>\n3\n(').encode()
                    if fmt == 'binary':
                        payload = expected.astype(dtype).tobytes()
                    else:
                        payload = ('\n'+'\n'.join('('+' '.join(map(str, row))+')' if vector else str(row[0])
                                                  for row in expected)+'\n').encode()
                    raw = header+payload+b');\nboundaryField { outlet { value uniform 99; } }'
                    for compressed in (False, True):
                        path = self.path/('field.gz' if compressed else 'field')
                        path.write_bytes(gzip.compress(raw) if compressed else raw)
                        with self.subTest(vector=vector, fmt=fmt, dtype=dtype, gzip=compressed):
                            np.testing.assert_array_equal(b.read_internal_field(path).values, expected)

    def test_uniform_vector(self):
        path = self.path/'U'
        path.write_text('FoamFile {format ascii; class volVectorField;} internalField uniform (1 2 3);')
        np.testing.assert_array_equal(b.read_internal_field(path).values, [[1, 2, 3]])

    def test_internal_and_old_time_selection(self):
        for phase, fraction in zip(b.PHASES, (.2, .3, .5)):
            raw = f'FoamFile {{format ascii; class volScalarField;}} internalField uniform {fraction};'
            (self.path/f'alpha.{phase}').write_text(raw)
            (self.path/f'alpha.{phase}_0').write_text(raw)
        result = b.field_health_report(self.path)
        self.assertEqual(result['alpha_sum']['max_abs_error_from_one'], 0)
        self.assertTrue(result['alpha_sum']['complete'])
        self.assertEqual(len(result['alpha_sum']['fields']), 3)
        (self.path/'alpha.air').unlink()
        self.assertFalse(b.field_health_report(self.path)['alpha_sum']['complete'])

    def test_zero_norm_and_bad_payload(self):
        zero = self.path/'zero'; one = self.path/'one'
        for path, value in ((zero, 0), (one, 1)):
            path.write_text(f'FoamFile {{format ascii; class volScalarField;}} internalField uniform {value};')
        comparison = b.compare_fields(zero, one)
        self.assertIsNone(comparison['rel_l2'])
        self.assertEqual(comparison['max_abs'], 1)
        bad = self.path/'bad'
        bad.write_bytes(b'FoamFile {format binary; class volScalarField;} internalField nonuniform List<scalar> 2 (abc);')
        with self.assertRaises(b.BenchmarkError):
            b.read_internal_field(bad)


if __name__ == '__main__':
    unittest.main()
