# SPDX-License-Identifier: GPL-3.0-or-later
"""ctypes bridge for sharp implicit cell geometry; no shape data enters it."""
import ctypes as C
from pathlib import Path
import numpy as np

class SharpIntegrator:
    def __init__(self, library, backend=0):
        self.lib=C.CDLL(str(Path(library).resolve()));self.backend=backend
        self.fn=self.lib.implicit_integrals
        self.fn.argtypes=[C.c_int,C.c_int,C.POINTER(C.c_double),C.c_size_t,
                          C.POINTER(C.c_int64),C.c_double,C.c_uint,
                          C.POINTER(C.c_double),C.POINTER(C.c_double),C.POINTER(C.c_uint32)]
        self.fn.restype=C.c_int
        self.facefn=self.lib.implicit_faces
        self.facefn.argtypes=[C.c_int,C.c_int,C.POINTER(C.c_double),C.c_size_t,
                              C.POINTER(C.c_int64),C.c_double,C.c_uint,
                              C.POINTER(C.c_double),C.POINTER(C.c_uint32)]
        self.facefn.restype=C.c_int

    def faces(self, coefficients, faces, tolerance=1e-7, depth=10):
        coefficients=np.ascontiguousarray(coefficients,dtype=np.float64)
        n=coefficients.shape[0]-3
        if coefficients.shape!=(n+3,)*3:raise ValueError('non-cubic coefficient layout')
        faces=np.ascontiguousarray(faces,dtype=np.int64)
        geometry=np.full((len(faces),9),np.nan)
        status=np.full(len(faces),0xffffffff,dtype=np.uint32)
        d=lambda x:x.ctypes.data_as(C.POINTER(C.c_double))
        code=self.facefn(self.backend,n,d(coefficients),len(faces),
                         faces.ctypes.data_as(C.POINTER(C.c_int64)),tolerance,depth,d(geometry),
                         status.ctypes.data_as(C.POINTER(C.c_uint32)))
        if code:raise RuntimeError(f'implicit face backend {self.backend}: {code}')
        return geometry,status

    def evaluate(self, coefficients, cells, tolerance=1e-7, depth=10, jacobian=False):
        coefficients=np.ascontiguousarray(coefficients,dtype=np.float64)
        n=coefficients.shape[0]-3
        if coefficients.shape!=(n+3,)*3:raise ValueError('non-cubic coefficient layout')
        cells=np.ascontiguousarray(cells,dtype=np.int64)
        geometry=np.full((len(cells),13),np.nan)
        jac=np.full((len(cells),64),np.nan) if jacobian else None
        status=np.full(len(cells),0xffffffff,dtype=np.uint32)
        d=lambda x:x.ctypes.data_as(C.POINTER(C.c_double)) if x is not None else None
        code=self.fn(self.backend,n,d(coefficients),len(cells),
                     cells.ctypes.data_as(C.POINTER(C.c_int64)),tolerance,depth,d(geometry),d(jac),
                     status.ctypes.data_as(C.POINTER(C.c_uint32)))
        if code:raise RuntimeError(f'implicit integrator backend {self.backend}: {code}')
        return geometry,jac,status
