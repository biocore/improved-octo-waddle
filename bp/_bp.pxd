cimport numpy as np
cimport cython
from cpython cimport bool

@cython.final
cdef class BPNode:
    cdef:
        public unicode name
        public np.double_t length

@cython.final
cdef class BP:
    cdef:
        public np.ndarray B 
        np.ndarray _r_index, _k_index_0, _k_index_1, _e_index
        np.ndarray _closeopen_index, _names, _lengths
# pxd is not inline with pyx...
    cpdef inline unicode name(self, Py_ssize_t i)
    cpdef inline np.double_t length(self, Py_ssize_t i)
    cpdef inline BPNode get_node(self, Py_ssize_t i)
    cpdef inline np.uint32_t rank(self, Py_ssize_t t, Py_ssize_t i)
    cpdef inline np.uint32_t select(self, Py_ssize_t t, Py_ssize_t k)
    cdef inline np.uint32_t _excess(self, Py_ssize_t i)
    cpdef inline np.uint32_t excess(self, Py_ssize_t i)
    cpdef inline np.int32_t fwdsearch(self, Py_ssize_t i, int d)
    cpdef inline Py_ssize_t bwdsearch(self, Py_ssize_t i, int d)
    cpdef inline np.int32_t close(self, Py_ssize_t i)
    cpdef inline np.int32_t open(self, Py_ssize_t i)
    cpdef inline np.int32_t enclose(self, Py_ssize_t i)
    cpdef np.uint32_t rmq(self, Py_ssize_t i, Py_ssize_t j)
    cpdef np.uint32_t rMq(self, Py_ssize_t i, Py_ssize_t j)
    cpdef inline np.uint8_t isleaf(self, Py_ssize_t i)
    cpdef inline np.uint32_t postorderselect(self, Py_ssize_t k)
    cpdef BP shear(self, np.ndarray[np.uint32_t, ndim=1] tips)
    cpdef BP collapse(self)
    cdef BP _mask_from_self(self, np.ndarray[np.uint8_t, ndim=1] mask, np.ndarray[np.double_t, ndim=1] lengths)
    cdef inline void _set_closeopen_cache(self)
    cpdef inline np.uint32_t ntips(self)
