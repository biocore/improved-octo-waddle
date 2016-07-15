cimport numpy as np
cimport cython

ctypedef np.npy_intp SIZE_t
ctypedef np.npy_uint32 UINT32_t
ctypedef np.npy_float64 DOUBLE_t
ctypedef np.npy_uint8 BOOL_t


@cython.final
cdef class BPNode:
    cdef:
        public unicode name
        public DOUBLE_t length

@cython.final
cdef class BP:
    cdef:
        public np.ndarray B 
        np.ndarray _r_index_0, _r_index_1, _k_index_0, _k_index_1, _e_index
        np.ndarray _closeopen_index, _names, _lengths
        SIZE_t[:, :] _rmm
        SIZE_t size

    cpdef inline unicode name(self, SIZE_t i)
    cpdef inline DOUBLE_t length(self, SIZE_t i)
    cpdef inline BPNode get_node(self, SIZE_t i)
    cpdef inline SIZE_t rank(self, SIZE_t t, SIZE_t i) nogil
    cpdef inline SIZE_t select(self, SIZE_t t, SIZE_t k) nogil
    cdef inline  SIZE_t _excess(self, SIZE_t i) 
    cpdef inline SIZE_t excess(self, SIZE_t i) nogil
    cpdef inline SIZE_t fwdsearch_rmm(self, SIZE_t i, int d) nogil
    cpdef inline SIZE_t fwdsearch(self, SIZE_t i, int d) nogil
    cpdef inline SIZE_t bwdsearch_rmm(self, SIZE_t i, int d) nogil
    cpdef inline SIZE_t bwdsearch(self, SIZE_t i, int d) nogil
    cpdef inline SIZE_t close(self, SIZE_t i) nogil
    cpdef inline SIZE_t open(self, SIZE_t i) nogil
    cpdef inline SIZE_t enclose(self, SIZE_t i) nogil
    cpdef SIZE_t rmq(self, SIZE_t i, SIZE_t j)
    cpdef SIZE_t rMq(self, SIZE_t i, SIZE_t j)
    cpdef inline BOOL_t isleaf(self, SIZE_t i) nogil
    cpdef inline SIZE_t postorderselect(self, SIZE_t k) nogil
    cpdef inline SIZE_t preorderselect(self, SIZE_t k) nogil
    cpdef BP shear(self, set tips)
    cpdef BP collapse(self)
    cdef BP _mask_from_self(self, np.ndarray[BOOL_t, ndim=1] mask, np.ndarray[DOUBLE_t, ndim=1] lengths)
    cdef inline void _set_closeopen_cache(self)
    cpdef inline SIZE_t ntips(self) nogil
    cpdef SIZE_t nsibling(self, SIZE_t i) nogil
    cpdef SIZE_t psibling(self, SIZE_t i) nogil
    cpdef SIZE_t lchild(self, SIZE_t i) nogil
    cpdef SIZE_t fchild(self, SIZE_t i) nogil
    cpdef SIZE_t parent(self, SIZE_t i) nogil
    cpdef SIZE_t depth(self, SIZE_t i) nogil
    cpdef SIZE_t root(self) nogil
