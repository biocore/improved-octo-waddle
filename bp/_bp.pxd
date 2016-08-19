cimport numpy as np
cimport cython

from bp._ba cimport BIT_ARRAY

ctypedef np.npy_intp SIZE_t
ctypedef np.npy_uint32 UINT32_t
ctypedef np.npy_float64 DOUBLE_t
ctypedef np.npy_uint8 BOOL_t

cdef class mM:
    cdef int b  # block size
    cdef int n_tip  # number of tips in the binary tree
    cdef int n_internal  # number of internal nodes in the binary tree
    cdef int n_total  # total number of nodes in the binary tree
    cdef int height  # the height of the binary tree
    cdef int m_idx  # m is minimum excess
    cdef int M_idx  # M is maximum excess
    cdef int r_idx  # rank
    cdef int k0_idx
    cdef SIZE_t[:, ::1] mM

    cdef void rmm(self, BOOL_t[:] B, int B_size) nogil

@cython.final
cdef class BPNode:
    cdef:
        public unicode name
        public DOUBLE_t length

@cython.final
cdef class BP:
    cdef:
        public np.ndarray B 
        BOOL_t* _b_ptr
        #SIZE_t[:] _r_index_0
        #SIZE_t[:] _r_index_1
        SIZE_t[:] _k_index_0
        SIZE_t[:] _k_index_1 
        SIZE_t[:] _e_index
        np.ndarray _names
        np.ndarray _lengths
        mM _rmm
        SIZE_t size

    cpdef inline unicode name(self, SIZE_t i)
    cpdef inline DOUBLE_t length(self, SIZE_t i)
    cpdef inline BPNode get_node(self, SIZE_t i)
    #cdef inline SIZE_t rank_rmm(self, SIZE_t t, SIZE_t i) nogil
    cdef inline SIZE_t rank(self, SIZE_t t, SIZE_t i) nogil
    cpdef inline SIZE_t select(self, SIZE_t t, SIZE_t k) nogil
    cpdef inline SIZE_t select_rmm(self, SIZE_t t, SIZE_t k)
    cdef SIZE_t _excess(self, SIZE_t i) nogil
    cdef SIZE_t excess(self, SIZE_t i) nogil
    cdef SIZE_t fwdsearch(self, SIZE_t i, int d) nogil
    cdef SIZE_t bwdsearch(self, SIZE_t i, int d) nogil
    cdef SIZE_t fwdsearch_naive(self, SIZE_t i, int d) nogil
    cdef SIZE_t bwdsearch_naive(self, SIZE_t i, int d) nogil
    cdef inline SIZE_t close(self, SIZE_t i) nogil
    cdef inline SIZE_t open(self, SIZE_t i) nogil
    cdef inline BOOL_t isleaf(self, SIZE_t i) nogil
    
    cdef inline SIZE_t enclose(self, SIZE_t i) nogil
    cpdef SIZE_t rmq(self, SIZE_t i, SIZE_t j)
    cpdef SIZE_t rMq(self, SIZE_t i, SIZE_t j)
    cpdef inline SIZE_t postorderselect(self, SIZE_t k) nogil
    cpdef inline SIZE_t preorderselect(self, SIZE_t k) nogil
    cpdef BP shear(self, set tips)
    cpdef BP collapse(self)
    cdef BP _mask_from_self(self, BIT_ARRAY* mask, np.ndarray[DOUBLE_t, ndim=1] lengths)
    cpdef SIZE_t ntips(self) nogil
    cdef SIZE_t nsibling(self, SIZE_t i) nogil
    cdef SIZE_t psibling(self, SIZE_t i) nogil
    cdef SIZE_t lchild(self, SIZE_t i) nogil
    cdef SIZE_t fchild(self, SIZE_t i) nogil
    cdef SIZE_t parent(self, SIZE_t i) nogil
    cdef SIZE_t depth(self, SIZE_t i) nogil
    cdef SIZE_t root(self) nogil
    cdef int scan_block_forward(self, int i, int k, int b, int d) nogil
    cdef int scan_block_backward(self, int i, int k, int b, int d) nogil
    cdef DOUBLE_t unweighted_unifrac(self, SIZE_t[:] u, SIZE_t[:] v) nogil
