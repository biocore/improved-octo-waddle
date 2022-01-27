cimport numpy as np
cimport cython

from bp._ba cimport BIT_ARRAY

ctypedef np.npy_intp SIZE_t
ctypedef np.npy_uint32 UINT32_t
ctypedef np.npy_int32 INT32_t
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
    cdef SIZE_t[:, ::1] mM
    cdef SIZE_t[:] r

    cdef void rmm(self, BOOL_t[:] B, int B_size) nogil


@cython.final
cdef class BP:
    cdef:
        public np.ndarray B 
        BOOL_t* _b_ptr
        SIZE_t[:] _e_index
        SIZE_t[:] _k_index_0
        SIZE_t[:] _k_index_1
        np.ndarray _names
        np.ndarray _lengths
        np.ndarray _edges
        np.ndarray _edge_lookup
        mM _rmm
        SIZE_t size

    cdef inline SIZE_t rank(self, SIZE_t t, SIZE_t i) nogil
    cdef inline SIZE_t select(self, SIZE_t t, SIZE_t k) nogil
    cdef SIZE_t _excess(self, SIZE_t i) nogil
    cdef SIZE_t excess(self, SIZE_t i) nogil
    cdef SIZE_t fwdsearch(self, SIZE_t i, int d) nogil
    cdef SIZE_t bwdsearch(self, SIZE_t i, int d) nogil
    cpdef inline SIZE_t close(self, SIZE_t i) nogil
    cdef inline SIZE_t open(self, SIZE_t i) nogil
    cpdef inline BOOL_t isleaf(self, SIZE_t i) nogil
    cdef inline SIZE_t enclose(self, SIZE_t i) nogil
    cdef BP _mask_from_self(self, BIT_ARRAY* mask, np.ndarray[DOUBLE_t, ndim=1] lengths)
    cpdef SIZE_t nsibling(self, SIZE_t i) nogil
    cpdef SIZE_t psibling(self, SIZE_t i) nogil
    cpdef SIZE_t lchild(self, SIZE_t i) nogil
    cpdef SIZE_t fchild(self, SIZE_t i) nogil
    cpdef SIZE_t parent(self, SIZE_t i) nogil
    cpdef SIZE_t depth(self, SIZE_t i) nogil
    cpdef SIZE_t root(self) nogil
    cdef int scan_block_forward(self, int i, int k, int b, int d) nogil
    cdef int scan_block_backward(self, int i, int k, int b, int d) nogil
    cdef void _set_edges(self, np.ndarray[INT32_t, ndim=1] edges)

    # TODO: evalute down the road what methods should be cdef. There is a 
    # performance cost for cpdef, so for high use functions, it may make sense
    # to punt down to cdef.
    # http://notes-on-cython.readthedocs.io/en/latest/fibo_speed.html
    cpdef inline unicode name(self, SIZE_t i)
    cpdef inline DOUBLE_t length(self, SIZE_t i)
    cpdef inline INT32_t edge(self, SIZE_t i)
    cpdef SIZE_t edge_from_number(self, INT32_t n)
    cpdef SIZE_t rmq(self, SIZE_t i, SIZE_t j) nogil
    cpdef SIZE_t rMq(self, SIZE_t i, SIZE_t j) nogil
    cpdef SIZE_t postorderselect(self, SIZE_t k) nogil
    cpdef SIZE_t postorder(self, SIZE_t i) nogil
    cpdef SIZE_t preorderselect(self, SIZE_t k) nogil
    cpdef SIZE_t preorder(self, SIZE_t i) nogil
    cpdef BOOL_t isancestor(self, SIZE_t i, SIZE_t j) nogil
    cpdef SIZE_t levelancestor(self, SIZE_t i, SIZE_t d) nogil
    cpdef SIZE_t subtree(self, SIZE_t i) nogil
    cpdef BP shear(self, set tips)
    cpdef BP collapse(self)
    cpdef SIZE_t ntips(self) nogil
    cpdef SIZE_t levelnext(self, SIZE_t i) nogil
    cpdef SIZE_t height(self, SIZE_t i) nogil
    cpdef SIZE_t deepestnode(self, SIZE_t i) nogil
    cpdef SIZE_t lca(self, SIZE_t i, SIZE_t j) nogil
