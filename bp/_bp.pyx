#cython: boundscheck=False, wraparound=False, cdivision=True
# ----------------------------------------------------------------------------
# Copyright (c) 2013--, BP development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

### NOTE: some doctext strings are copied and pasted from manuscript
### http://www.dcc.uchile.cl/~gnavarro/ps/tcs16.2.pdf

from libc.math cimport ceil, log as ln, pow, log2

import numpy as np
cimport numpy as np
cimport cython

from bp._binary_tree cimport * #bt_node_from_left, bt_left_child, bt_right_child

np.import_array()

cdef extern from "limits.h":
    int INT_MAX

DOUBLE = np.float64
SIZE = np.intp
BOOL = np.uint8


cdef inline int min(int a, int b) nogil:
    if a > b:
        return b
    else:
        return a


cdef inline int max(int a, int b) nogil:
    if a > b:
        return a
    else:
        return b


@cython.final
cdef class BPNode:
    """A version of a node

    Attributes
    ----------
    name : unicode
        The name of the node
    length : np.double_t
        A branch length from this node to a parent
    """
    def __cinit__(self, unicode name, DOUBLE_t length):
        self.name = name
        self.length = length


@cython.final
cdef class BP:
    """A balanced parentheses succinct data structure tree representation

    The basis for this implementation is the data structure described by
    Cordova and Navarro [1]. In some instances, some docstring text was copied
    verbatim from the manuscript. This does not implement the bucket-based
    trees, although that would be a very interesting next step. 

    A node in this data structure is represented by 2 bits, an open parenthesis
    and a close parenthesis. The implementation uses a numpy uint8 type where
    an open parenthesis is a 1 and a close is a 0. In general, operations on
    this tree are best suited for passing in the opening parenthesis index, so
    for instance, if you'd like to use BP.isleaf to determine if a node is a 
    leaf, the operation is defined only for using the opening parenthesis. At 
    this time, there is some ambiguity over what methods can handle a closing
    parenthesis.

    Node attributes, such as names, are stored external to this data structure.

    The motivator for this data structure is pure performance both in space and
    time. As such, there is minimal sanity checking. It is advised to use this
    structure with care, and ideally within a framework which can assure 
    sanity. 

    References
    ----------
    [1] http://www.dcc.uchile.cl/~gnavarro/ps/tcs16.2.pdf
    """

    def __cinit__(self, np.ndarray[BOOL_t, ndim=1] B,
                  np.ndarray[SIZE_t, ndim=1] closeopen=None,
                  np.ndarray[object, ndim=1] names=None,
                  np.ndarray[DOUBLE_t, ndim=1] lengths=None):
        cdef:
            SIZE_t i, size
            np.ndarray[SIZE_t, ndim=1] _e_index, _k_index_0, _k_index_1
            np.ndarray[SIZE_t, ndim=1] _closeopen_index
            np.ndarray[object, ndim=1] _names
            np.ndarray[DOUBLE_t, ndim=1] _lengths
            np.ndarray[SIZE_t, ndim=1] _r_index_0, _r_index_1
            np.ndarray[SIZE_t, ndim=2] _rmm
            SIZE_t* excess_ptr

        # the tree is only valid if it is balanaced
        assert B.sum() == (float(B.size) / 2)
        self.B = B
        self.size = B.size

        self._rmm = rmm(B, B.size)

        if names is not None:
            self._names = names
        else:
            self._names = np.full(self.B.size, None, dtype=object)

        if lengths is not None:
            self._lengths = lengths
        else:
            self._lengths = np.zeros(self.B.size, dtype=DOUBLE)

        # construct a rank index. These operations are performed frequently,
        # and easy to cache at a relatively minor memory expense
        _r_index_0 = np.cumsum((1 - B), dtype=SIZE)
        _r_index_1 = np.cumsum(B, dtype=SIZE)
        self._r_index_0 = _r_index_0
        self._r_index_1 = _r_index_1

        # construct a select index. These operations are performed frequently,
        # and easy to cache at a relatively minor memory expense. It cannot be
        # assumed that open and close will be same length so can't stack
        _k_index_0 = np.unique(_r_index_0,
                               return_index=True)[1].astype(SIZE)
        self._k_index_0 = _k_index_0
        _k_index_1 = np.unique(_r_index_1,
                               return_index=True)[1].astype(SIZE)
        self._k_index_1 = _k_index_1

        # construct an excess index. These operations are performed a lot, and
        # similarly can to rank and select, can be cached at a minimal expense.
        _e_index = np.empty(B.size, dtype=SIZE)
        excess_ptr = <SIZE_t*>_e_index.data
        for i in range(B.size):
            excess_ptr[i] = self._excess(i)
        self._e_index = _e_index

        # The closeopen index is not provided at construction as it can be 
        # determined at parse with very minimal overhead. 
        if closeopen is not None:
            self._closeopen_index = closeopen
        else:
            self._set_closeopen_cache()

    def write(self, object fname):
        np.savez_compressed(fname, names=self._names, lengths=self._lengths, 
                            B=self.B)

    @staticmethod 
    def read(object fname):
        data = np.load(fname)
        bp = BP(data['B'], names=data['names'], lengths=data['lengths'])
        return bp

    def set_names(self, np.ndarray[object, ndim=1] names):
        self._names = names

    def set_lengths(self, np.ndarray[DOUBLE_t, ndim=1] lengths):
        self._lengths = lengths

    cpdef inline unicode name(self, SIZE_t i):
        return self._names[i]

    cpdef inline DOUBLE_t length(self, SIZE_t i):
        cdef DOUBLE_t* length_ptr = <DOUBLE_t*>self._lengths.data
        return length_ptr[i]

    cpdef inline BPNode get_node(self, SIZE_t i):
        cdef DOUBLE_t* length_ptr = <DOUBLE_t*>self._lengths.data

        # might be possible to do a ptr to _names, see
        # https://github.com/h5py/h5py/blob/master/h5py/_conv.pyx#L177
        return BPNode(self._names[i], length_ptr[i])

    cdef inline void _set_closeopen_cache(self):
        cdef:
            SIZE_t i, j, n, m
            np.ndarray[SIZE_t, ndim=1] closeopen
            BOOL_t* b_ptr = <BOOL_t*> self.B.data
            SIZE_t* closeopen_ptr
        
        n = self.size

        closeopen = np.zeros(n, dtype=SIZE)
        closeopen_ptr = <SIZE_t*>closeopen.data

        for i in range(n):
            # if we haven't already cached it (cheaper than open/close call)
            # note: idx 0 is valid, but it will be on idx -1 and correspond to
            # root so this is safe.
            if closeopen_ptr[i] == 0:
                if b_ptr[i]:
                    j = self.fwdsearch(i, -1)
                else:
                    j = self.bwdsearch(i, 0) + 1

                closeopen_ptr[i] = j
                closeopen_ptr[j] = i

        self._closeopen_index = closeopen

    cpdef inline SIZE_t rank(self, SIZE_t t, SIZE_t i) nogil:
        """The number of occurrences of the bit t in B"""
        cdef SIZE_t* _r_index_1_ptr = <SIZE_t*> self._r_index_1.data
        cdef SIZE_t* _r_index_0_ptr = <SIZE_t*> self._r_index_0.data

        if t:
            return _r_index_1_ptr[i]
        else:
            return _r_index_0_ptr[i]

    cpdef inline SIZE_t select(self, SIZE_t t, SIZE_t k) nogil:
        """The position in B of the kth occurrence of the bit t."""
        cdef SIZE_t* _k_index_1_ptr = <SIZE_t*> self._k_index_1.data
        cdef SIZE_t* _k_index_0_ptr = <SIZE_t*> self._k_index_0.data

        if t:
            return _k_index_1_ptr[k]
        else:
            return _k_index_0_ptr[k]

    cdef inline SIZE_t _excess(self, SIZE_t i):
        """Actually compute excess"""
        if i < 0:
            return 0  # wasn't stated as needed but appears so given testing
        return (2 * self.rank(1, i) - i) - 1

    cpdef inline SIZE_t excess(self, SIZE_t i) nogil:
        """the number of opening minus closing parentheses in B[1, i]"""
        # same as: self.rank(1, i) - self.rank(0, i)
        cdef SIZE_t* _e_index_ptr = <SIZE_t*> self._e_index.data
        return _e_index_ptr[i]
    
    cpdef inline SIZE_t fwdsearch(self, SIZE_t i, int d) nogil:
        """Forward search for excess by depth"""
        cdef:
            SIZE_t j, n = self.size
            SIZE_t b
            SIZE_t* e_index_ptr = <SIZE_t*> self._e_index.data

        b = e_index_ptr[i] + d

        for j in range(i + 1, n):
            if e_index_ptr[j] == b:
                return j

        return -1  # wasn't stated as needed but appears so given testing

    cpdef inline SIZE_t fwdsearch_rmm(self, SIZE_t i, int d) nogil:
        return fwdsearch(self, self._rmm, i, d)

    cpdef inline SIZE_t bwdsearch(self, SIZE_t i, int d) nogil:
        """Backward search for excess by depth"""
        cdef:
            SIZE_t j
            SIZE_t b 
            SIZE_t* e_index_ptr = <SIZE_t*> self._e_index.data

        b = e_index_ptr[i] + d

        for j in range(i - 1, -1, -1):
            if e_index_ptr[j] == b:
                return j

        return -1

    cpdef inline SIZE_t bwdsearch_rmm(self, SIZE_t i, int d) nogil:
        return bwdsearch(self, self._rmm, i, d)

    cpdef inline SIZE_t close(self, SIZE_t i) nogil:
        """The position of the closing parenthesis that matches B[i]"""
        cdef BOOL_t* b_ptr = <BOOL_t*> self.B.data
        cdef SIZE_t* co_ptr = <SIZE_t*> self._closeopen_index.data

        if not b_ptr[i]:
            # identity: the close of a closed parenthesis is itself
            return i

        return co_ptr[i]

    cpdef inline SIZE_t open(self, SIZE_t i) nogil:
        """The position of the opening parenthesis that matches B[i]"""
        cdef BOOL_t* b_ptr = <BOOL_t*> self.B.data
        cdef SIZE_t* co_ptr = <SIZE_t*> self._closeopen_index.data

        if b_ptr[i] or i <= 0:
            # identity: the open of an open parenthesis is itself
            # the open of 0 is open. A negative index cannot be open, so just return
            return i

        return co_ptr[i]

    cpdef inline SIZE_t enclose(self, SIZE_t i) nogil:
        """The opening parenthesis of the smallest matching pair that contains position i"""
        cdef BOOL_t* b_ptr = <BOOL_t*> self.B.data
        if b_ptr[i]:
            return self.bwdsearch(i, -2) + 1
        else:
            return self.bwdsearch(i - 1, -2) + 1

    cpdef SIZE_t rmq(self, SIZE_t i, SIZE_t j):
        """The leftmost minimum excess in i -> j"""
        cdef:
            SIZE_t k, min_k
            SIZE_t min_v, obs_v

        min_k = i
        min_v = self.excess(i)  # a value larger than what will be tested
        for k in range(i, j + 1):
            obs_v = self.excess(k)
            if obs_v < min_v:
                min_k = k
                min_v = obs_v
        return min_k

    cpdef SIZE_t rMq(self, SIZE_t i, SIZE_t j):
        """The leftmost maximmum excess in i -> j"""
        cdef:
            SIZE_t k, max_k
            SIZE_t max_v, obs_v

        max_k = i
        max_v = self.excess(i)  # a value larger than what will be tested
        for k in range(i, j + 1):
            obs_v = self.excess(k)
            if obs_v > max_v:
                max_k = k
                max_v = obs_v

        return max_k

        # return np.array([self.excess(k) for k in range(i, j + 1)]).argmax() + i

    def __reduce__(self):
        return (BP, (self.B, self._closeopen_index, self._names, self._lengths))

    cpdef SIZE_t depth(self, SIZE_t i) nogil:
        """The depth of node i"""
        cdef SIZE_t* _e_index_ptr = <SIZE_t*> self._e_index.data
        return _e_index_ptr[i]

    cpdef SIZE_t root(self) nogil:
        """The root of the tree"""
        return 0

    cpdef SIZE_t parent(self, SIZE_t i) nogil:
        """The parent of node i"""
        return self.enclose(i)

    cpdef inline BOOL_t isleaf(self, SIZE_t i) nogil:
        """Whether the node is a leaf"""
        # publication describe this operation as "iff B[i+1] == 0" which is incorrect

        # most likely there is an implicit conversion here
        cdef BOOL_t* B = <BOOL_t*> self.B.data

        return B[i] and (not B[i + 1])

    cpdef SIZE_t fchild(self, SIZE_t i) nogil:
        """The first child of i (i.e., the left child)

        fchild(i) = i + 1 (if i is not a leaf)

        Returns 0 if the node is a leaf as the root cannot be a child by
        definition.
        """
        cdef BOOL_t* b_ptr = <BOOL_t*> self.B.data
        if b_ptr[i]:
            if self.isleaf(i):
                return 0
            else:
                return i + 1
        else:
            return self.fchild(self.open(i))

    cpdef SIZE_t lchild(self, SIZE_t i) nogil:
        """The last child of i (i.e., the right child)

        lchild(i) = open(close(i) − 1) (if i is not a leaf)

        Returns 0 if the node is a leaf as the root cannot be a child by
        definition.
        """
        cdef BOOL_t* b_ptr = <BOOL_t*> self.B.data
        if b_ptr[i]:
            if self.isleaf(i):
                return 0
            else:
                return self.open(self.close(i) - 1)
        else:
            return self.lchild(self.open(i))

    def mincount(self, i, j):
        """number of occurrences of the minimum in excess(i), excess(i + 1), . . . , excess(j)."""
        excess, counts = np.unique([self.excess(k) for k in range(i, j + 1)], return_counts=True)
        return counts[excess.argmin()]

    def minselect(self, i, j, q):
        """position of the qth minimum in excess(i), excess(i + 1), . . . , excess(j)."""
        counts = np.array([self.excess(k) for k in range(i, j + 1)])
        index = counts == counts.min()

        if index.sum() < q:
            return None
        else:
            return i + index.nonzero()[0][q - 1]

    cpdef SIZE_t nsibling(self, SIZE_t i) nogil:
        """The next sibling of i (i.e., the sibling to the right)

        nsibling(i) = close(i) + 1 (if the result j holds B[j] = 0 then i has no next sibling)

        Will return 0 if there is no sibling. This makes sense as the root
        cannot have a sibling by definition
        """
        cdef:
            SIZE_t pos
        cdef BOOL_t* b_ptr = <BOOL_t*> self.B.data

        if b_ptr[i]:
            pos = self.close(i) + 1
        else:
            pos = self.nsibling(self.open(i))

        if pos >= self.size:
            return 0
        elif b_ptr[pos]:
            return pos
        else:
            return 0

    cpdef SIZE_t psibling(self, SIZE_t i) nogil:
        """The previous sibling of i (i.e., the sibling to the left)

        psibling(i) = open(i − 1) (if B[i − 1] = 1 then i has no previous sibling)

        Will return 0 if there is no sibling. This makes sense as the root
        cannot have a sibling by definition
        """
        cdef:
            np.uint32_t pos
        cdef BOOL_t* b_ptr = <BOOL_t*> self.B.data

        if b_ptr[i]:
            if b_ptr[max(0, i - 1)]:
                return 0

            pos = self.open(i - 1)
        else:
            pos = self.psibling(self.open(i))

        if pos < 0:
            return 0
        elif b_ptr[pos]:
            return pos
        else:
            return 0

    def preorder(self, i):
        """Preorder rank of node i"""
        # preorder(i) = rank1(i),
        if self.B[i]:
            return self.rank(1, i)
        else:
            return self.preorder(self.open(i))

    cpdef inline SIZE_t preorderselect(self, SIZE_t k) nogil:
        """The node with preorder k"""
        # preorderselect(k) = select1(k),
        return self.select(1, k)

    def postorder(self, i):
        """Postorder rank of node i"""
        # postorder(i) = rank0(close(i)),
        if self.B[i]:
            return self.rank(0, self.close(i))
        else:
            return self.rank(0, i)

    cpdef inline SIZE_t postorderselect(self, SIZE_t k) nogil:
        """The node with postorder k"""
        # postorderselect(k) = open(select0(k)),
        return self.open(self.select(0, k))

    def isancestor(self, i, j):
        """Whether i is an ancestor of j"""
        # isancestor(i, j) iff i ≤ j < close(i)
        if i == j:
            return False

        if not self.B[i]:
            i = self.open(i)

        return i <= j < self.close(i)

    def subtree(self, i):
        """The number of nodes in the subtree of i"""
        # subtree(i) = (close(i) − i + 1)/2.
        if not self.B[i]:
            i = self.open(i)

        return (self.close(i) - i + 1) / 2

    def levelancestor(self, i, d):
        """ancestor j of i such that depth(j) = depth(i) − d"""
        # levelancestor(i, d) = bwdsearch(i, −d−1)+1
        if d <= 0:
            return -1

        if not self.B[i]:
            i = self.open(i)

        return self.bwdsearch(i, -d - 1) + 1

    def levelnext(self, i):
        """The next node with the same depth"""
        # levelnext(i) = fwdsearch(close(i), 1)
        return self.fwdsearch(self.close(i), 1)

    def levelprev(self, i):
        """The previous node with the same depth"""
        #levelprev(i) = open(bwdsearch(i, 0)+1)
        j = self.open(self.bwdsearch(self.open(i), 0))# + 1
        print(i, self.excess(i), self.excess(j), self.depth(i), self.depth(j))
        #print(i, self.bwdsearch(i, 0), self.open(self.bwdsearch(i, 0)))
        #return self.bwdsearch(self.open(i), 0)
        return j #self.bwdsearch(i - 1)
        #if not self.B[i]:
        #    i = self.open(i)

        #j = self.open(self.bwdsearch(i, 0))
        #if j < 0:
        #    return j
        #else:
        #    return j #+ 1

    def levelleftmost(self, d):
        #levelleftmost(d) = fwdsearch(0, d),
        pass

    def levelrightmost(self, d):
        #levelrightmost(d) = open(bwdsearch(2n + 1, d)).
        pass

    def degree(self, i):
        #degree(i) = mincount(i + 1, close(i) − 1),
        pass

    def child(i, q):
        # child(i, q) = minselect(i+1, close(i)−1, q−1)+1 for q > 1
        # (for q = 1 it is fchild(i)),
        pass

    def childrank(self, i):
        # childrank(i) = mincount(parent(i) + 1, i) + 1
        # unless B[i − 1] = 1
        # (in which case childrank(i) = 1)
        pass

    def lca(self, i, j):
        # lca(i, j) = parent(rmq(i, j) + 1)
        # unless isancestor(i, j)
        # (so lca(i, j) = i) or isancestor(j, i) (so lca(i, j) = j),
        if self.isancestor(i, j):
            return i
        elif self.isancestor(j, i):
            return j
        else:
            return self.parent(self.rmq(i, j) + 1)

    def deepestnode(self, i):
        # deepestnode(i) = rMq(i, close(i)),
        return self.rMq(self.open(i), self.close(i))

    def height(self, i):
        """the height of i (distance to its deepest node)"""
        # height(i) = excess(deepestnode(i)) − excess(i).
        return self.excess(self.deepestnode(i)) - self.excess(self.open(i))

    cpdef BP shear(self, set tips):
        """Remove all nodes from the tree except tips and ancestors of tips

        Parameters
        ----------
        tips : set of str
            The set of tip names to retain

        Returns
        -------
        BP
            A new BP tree corresponding to only the described tips and their
            ancestors.
        """
        cdef:
            np.ndarray[SIZE_t, ndim=1] tip_indices
            SIZE_t i, n = len(tips)
            SIZE_t p, t
            np.ndarray[BOOL_t, ndim=1] mask
            BOOL_t* mask_ptr

        mask = np.zeros(self.B.size, dtype=BOOL)
        mask_ptr = <BOOL_t*>mask.data

        mask_ptr[self.root()] = 1
        mask_ptr[self.close(self.root())] = 1

        for i in range(self.B.size):
            # isleaf is only defined on the open parenthesis
            if self.isleaf(i):
                if self.name(i) in tips:  # gil is required for set operation
                    with nogil:
                        mask_ptr[i] = 1
                        mask_ptr[i + 1] = 1 # close

                        p = self.parent(i)
                        while p != 0 and mask_ptr[p] == 0:
                            mask_ptr[p] = 1
                            mask_ptr[self.close(p)] = 1

                            p = self.parent(p)

        return self._mask_from_self(mask, self._lengths)

    cdef BP _mask_from_self(self, np.ndarray[BOOL_t, ndim=1] mask, 
                            np.ndarray[DOUBLE_t, ndim=1] lengths):
        cdef:
            SIZE_t i, k, n = mask.size, mask_sum = mask.sum()
            np.ndarray[BOOL_t, ndim=1] new_b
            np.ndarray[object, ndim=1] new_names
            np.ndarray[object, ndim=1] names = self._names
            np.ndarray[DOUBLE_t, ndim=1] new_lengths
            BOOL_t* b_ptr
            BOOL_t* new_b_ptr
            BOOL_t* mask_ptr
            DOUBLE_t* lengths_ptr
            DOUBLE_t* new_lengths_ptr

        k = 0

        b_ptr = <BOOL_t*> self.B.data
        lengths_ptr = <DOUBLE_t*> lengths.data

        new_b = np.empty(mask_sum, dtype=BOOL)
        new_names = np.empty(mask_sum, dtype=object)
        new_lengths = np.empty(mask_sum, dtype=DOUBLE)

        new_b_ptr = <BOOL_t*> new_b.data
        new_lengths_ptr = <DOUBLE_t*> new_lengths.data

        mask_ptr = <BOOL_t*>mask.data

        for i in range(n):
            if mask_ptr[i]:
                new_b_ptr[k] = b_ptr[i]

                # since names is dtype=object, gil is required
                new_names[k] = names[i]
                new_lengths_ptr[k] = lengths_ptr[i]
                k += 1

        return BP(new_b, names=new_names, lengths=new_lengths)

    cpdef BP collapse(self):
        cdef:
            SIZE_t i, n = self.B.sum()
            SIZE_t current, first, last
            np.ndarray[BOOL_t, ndim=1] mask
            np.ndarray[DOUBLE_t, ndim=1] new_lengths
            BOOL_t* mask_ptr
            DOUBLE_t* new_lengths_ptr

        mask = np.zeros(self.B.size, dtype=BOOL)
        mask_ptr = <BOOL_t*>mask.data

        mask_ptr[self.root()] = 1
        mask_ptr[self.close(self.root())] = 1

        new_lengths = self._lengths.copy()
        new_lengths_ptr = <DOUBLE_t*>new_lengths.data

        with nogil:
            for i in range(n):
                current = self.preorderselect(i)

                if self.isleaf(current):
                    mask_ptr[current] = 1
                    mask_ptr[self.close(current)] = 1
                else:
                    first = self.fchild(current)
                    last = self.lchild(current)

                    if first == last:
                        new_lengths_ptr[first] = new_lengths_ptr[first] + \
                                new_lengths_ptr[current]
                    else:
                        mask_ptr[current] = 1
                        mask_ptr[self.close(current)] = 1

        return self._mask_from_self(mask, new_lengths)

    cpdef inline SIZE_t ntips(self) nogil:
        cdef:
            SIZE_t i = 0
            SIZE_t count = 0
            SIZE_t n = self.size
            BOOL_t* B_ptr

        B_ptr = <BOOL_t*>self.B.data
        while i < (n - 1):
            if B_ptr[i] and not B_ptr[i+1]:
                count += 1
                i += 1
            i += 1

        return count


cdef SIZE_t[:, :] rmm(BOOL_t[:] B, int B_size) nogil:
    """Construct the rmM tree based off of Navarro and Sadakane

    http://www.dcc.uchile.cl/~gnavarro/ps/talg12.pdf
    """
    cdef int b  # block size
    cdef int n_tip  # number of tips in the binary tree
    cdef int n_internal  # number of internal nodes in the binary tree
    cdef int n_total  # total number of nodes in the binary tree
    cdef int height  # the height of the binary tree
    cdef int i, j, lvl, pos  # for loop support
    cdef int offset  # tip offset in binary tree for a given parenthesis
    cdef int lower_limit  # the lower limit of the bucket a parenthesis is in
    cdef int upper_limit  # the upper limit of the bucket a parenthesis is in
    
    cdef SIZE_t[:, :] mM  # exact min/max per bucket
    cdef int m_idx = 0  # m is minimum excess
    cdef int M_idx = 1  # M is maximum excess
    cdef int min_ = 0 # m, temporary when computing relative
    cdef int max_ = 0 # M, temporary when computing relative
    cdef int excess = 0 # e, temporary when computing relative

    # build tip info
    b = <int>ceil(ln(<double> B_size) * ln(ln(<double> B_size)))

    # determine the number of nodes and height of the binary tree
    n_tip = <int>ceil(B_size / <double> b)
    height = <int>ceil(log2(n_tip))
    n_internal = <int>(pow(2, height)) - 1
    n_total = n_tip + n_internal

    with gil:
        # creation of a memoryview directly or via numpy requires the GIL:
        # http://stackoverflow.com/a/22238012
        mM = np.zeros((n_total, 2), dtype=SIZE)

    # annoying, cannot do step in range if step is not known at runtime
    # see https://github.com/cython/cython/pull/520
    # for i in range(0, B_size, b):
    # as a result, doing a custom range using a while loop
    # compute for tips of rmM tree
    i = 0
    while i < B_size:
        offset = i // b
        lower_limit = i
        upper_limit = min(i + b, B_size)
        min_ = INT_MAX
        max_ = 0

        for j in range(lower_limit, upper_limit):
            # G function, a +-1 method where if B[j] == 1 we +1, and if
            # B[j] == 0 we -1
            excess += -1 + (2 * B[j]) 

            if excess < min_:
                min_ = excess

            if excess > max_:
                max_ = excess

            # at the left bound of the bucket
        
        mM[offset + n_internal, m_idx] = min_
        mM[offset + n_internal, M_idx] = max_
        
        i += b

    # compute for internal nodes of rmM tree in reverse level order starting 
    # at the level above the tips
    for lvl in range(height - 1, -1, -1):
        num_curr_nodes = <int>pow(2, lvl)

        # for each node in the level
        for pos in range(num_curr_nodes):
            # obtain the node, and the index to its children
            node = bt_node_from_left(pos, lvl)
            lchild = bt_left_child(node)
            rchild = bt_right_child(node)
            
            if lchild >= n_total:
                continue

            elif rchild >= n_total:
                mM[node, m_idx] = mM[lchild, m_idx] 
                mM[node, M_idx] = mM[lchild, M_idx] 
            else:    
                mM[node, m_idx] = min(mM[lchild, m_idx], mM[rchild, m_idx])
                mM[node, M_idx] = max(mM[lchild, M_idx], mM[rchild, M_idx])

    return mM


cdef inline int scan_block_forward(BP bp, int i, int k, int b, int d) nogil:
    """Scan a block forward from i

    Parameters
    ----------
    bp : BP
        The tree
    i : int
        The index position to start from in the tree
    k : int
        The block to explore
    b : int
        The block size
    d : int
        The depth to search for

    Returns
    -------
    int
        The index position of the result. -1 is returned if a result is not
        found.
    """
    cdef int lower_bound
    cdef int upper_bound
    cdef int j

    # lower_bound is block boundary or right of i
    lower_bound = max(k, 0) * b
    lower_bound = max(i + 1, lower_bound)

    # upper_bound is block boundary or end of tree
    upper_bound = min((k + 1) * b, bp.size)

    for j in range(lower_bound, upper_bound):
        if bp.excess(j) == d:
            return j
    
    return -1


cdef inline int scan_block_backward(BP bp, int i, int k, int b, int d) nogil:
    """Scan a block backward from i

    Parameters
    ----------
    bp : BP
        The tree
    i : int
        The index position to start from in the tree
    k : int
        The block to explore
    b : int
        The block size
    d : int
        The depth to search for

    Returns
    -------
    int
        The index position of the result. -1 is returned if a result is not
        found.
    """
    cdef int lower_bound
    cdef int upper_bound
    cdef int j
    
    # i and k are currently needed to handle the situation where 
    # k_start < i < k_end. It should be possible to resolve using partial 
    # excess.

    # range stop is exclusive, so need to set "stop" at -1 of boundary
    lower_bound = max(k, 0) * b - 1  # is it possible for k to be < 0?
    
    # include the right most position of the k-1 block so we can identify
    # closures spanning blocks. Not positive if this is correct, however if the
    # block is "()((", and we're searching for the opening paired with ")", 
    # we need to go to evaluate the excess prior to the first "(", at least as
    # "open" is defined in Cordova and Navarro
    if lower_bound >= 0:
        lower_bound -= 1
    
    # upper bound is block boundary or left of i, whichever is less
    # note that this is an inclusive boundary since this is a backward search
    upper_bound = min((k + 1) * b, bp.size) - 1
    upper_bound = min(i - 1, upper_bound)
    
    if upper_bound <= 0:
        return -1

    for j in range(upper_bound, lower_bound, -1):
        if bp.excess(j) == d:
            return j

    return -1


cdef int fwdsearch(BP bp, SIZE_t[:, :] mM, int i, int d) nogil:
    """Search forward from i for desired excess

    Parameters
    ----------
    bp : BP
        The tree
    mM : 2D memoryview of SIZE_t
        The binary tree of min and max excess values
    i : int
        The index to search forward from
    d : int
        The excess difference to search for (relative to E[i])
    
    Returns
    -------
    int
        The index of the result, or -1 if no result was found
    """
    cdef int m_idx = 0  # m is minimum excess
    cdef int M_idx = 1  # M is maximum excess
    cdef int b  # the block size
    cdef int n_tip  # the number of tips in the binary tree
    cdef int height  # the height of the binary tree
    cdef int k  # the block being interrogated
    cdef int result  # the result of a scan within a block
    cdef int node  # the node within the binary tree being examined
    
    #TODO: stash binary tree details in a struct/object
    b = <int>ceil(ln(<double> bp.size) * ln(ln(<double> bp.size)))
    n_tip = <int>ceil(bp.size / <double> b)
    height = <int>ceil(log2(n_tip))

    # get the block of parentheses to check
    k = i // b  

    # desired excess
    d += bp.excess(i)

    # see if our result is in our current block
    result = scan_block_forward(bp, i, k, b, d)

    # determine which node our block corresponds too
    node = bt_node_from_left(k, height)
    
    # special case: check sibling
    if result == -1 and bt_is_left_child(node):
        node = bt_right_sibling(node)
        k = node - <int>(pow(2, height) - 1)
        result = scan_block_forward(bp, i, k, b, d)

        # reset node and k in the event that result == -1
        k = i // b
        node = bt_left_sibling(node)
    
    # if we do not have a result, we need to begin traversal of the tree
    if result == -1:
        # walk up the tree
        while not bt_is_root(node):
            if bt_is_left_child(node):
                node = bt_right_sibling(node)
                if mM[node, m_idx] <= d  <= mM[node, M_idx]:
                    break
            node = bt_parent(node)
        
        if not bt_is_root(node):
            # descend until we hit a leaf node
            while not bt_is_leaf(node, height):
                node = bt_left_child(node)

                # evaluate right, if not found, pick left
                if not (mM[node, m_idx] <= d <= mM[node, M_idx]):
                    node = bt_right_sibling(node)
        
        else:
            # no solution
            return -1

        # we have found a block with contains our solution. convert from the
        # node index back into the block index
        k = node - <int>(pow(2, height) - 1)

        # scan for a result using the original d
        result = scan_block_forward(bp, i, k, b, d)

    return result


cdef int bwdsearch(BP bp, SIZE_t[:,:] mM, int i, int d) nogil:
    """Search backward from i for desired excess

    Parameters
    ----------
    bp : BP
        The tree
    mM : 2D memoryview of int
        The binary tree of min and max excess values
    i : int
        The index to search forward from
    d : int
        The excess difference to search for (relative to E[i])
    
    Returns
    -------
    int
        The index of the result, or -1 if no result was found
    """
    cdef int m_idx = 0  # m is minimum excess
    cdef int M_idx = 1  # M is maximum excess
    cdef int b  # the block size
    cdef int n_tip  # the number of tips in the binary tree
    cdef int height  # the height of the binary tree
    cdef int k  # the block being interrogated
    cdef int result  # the result of a scan within a block
    cdef int node  # the node within the binary tree being examined
  
    #TODO: stash details in a struct/object
    b = <int>ceil(ln(<double> bp.size) * ln(ln(<double> bp.size)))
    n_tip = <int>ceil(bp.size / <double> b)
    height = <int>ceil(log2(n_tip))

    # get the block of parentheses to check
    k = i // b  
    
    # desired excess
    d += bp.excess(i)

    # see if our result is in our current block
    result = scan_block_backward(bp, i, k, b, d)

    # determine which node our block corresponds too
    node = bt_node_from_left(k, height)

    # special case: check sibling
    if result == -1 and bt_is_right_child(node):
        node = bt_left_sibling(node)
        k = node - <int>(pow(2, height) - 1)
        result = scan_block_backward(bp, i, k, b, d)
    
        # reset node and k in the event that result == -1
        k = i // b
        node = bt_right_sibling(node)
    
    # if we do not have a result, we need to begin traversal of the tree
    if result == -1:
        while not bt_is_root(node):
            # right nodes cannot contain the solution as we are searching left
            # As such, if we are the right node already, evaluate its sibling.
            if bt_is_right_child(node):
                node = bt_left_sibling(node)
                if mM[node, m_idx] <= d <= mM[node, M_idx]:
                    break
            
            # if we did not find a valid node, adjust for the relative
            # excess of the current node, and ascend to the parent
            node = bt_parent(node)

        # if we did not hit the root, then we have a possible solution
        if not bt_is_root(node):
            # descend until we hit a leaf node
            while not bt_is_leaf(node, height):
                node = bt_right_child(node)

                # evaluate right, if not found, pick left
                if not (mM[node, m_idx] <= d <= mM[node, M_idx]):
                    node = bt_left_sibling(node)

        else:
            # no solution
            return -1

        # we have found a block with contains our solution. convert from the
        # node index back into the block index
        k = node - <int>(pow(2, height) - 1)

        # scan for a result
        result = scan_block_backward(bp, i, k, b, d)
        
    return result

######################################################################
### cython cdef function test code        
######################################################################
def test_fwdsearch():
    from bp import parse_newick
    bp = parse_newick('((a,b,(c)),d,((e,f)));')
    enmM = rmm(bp.B, bp.B.size)

    # simulating close so only testing open parentheses. A "close" on a closed
    # parenthesis does not make sense, so the result is not useful.
    # In practice, an "close" method should ensure it is operating on a closed
    # parenthesis.
    # [(open_idx, close_idx), ...]
    exp = [(1, 10), (0, 21), (2, 3), (4, 5), (6, 9), (7, 8), (11, 12), 
           (13, 20), (14, 19), (15, 16), (17, 18)]

    for open_, exp_close in exp:
        obs_close = fwdsearch(bp, enmM, open_, -1)
        assert obs_close == exp_close

    # slightly modified version of fig2 with an extra child forcing a test
    # of the direct sibling check with negative partial excess

    # this translates into:
    # 012345678901234567890123
    # ((()()(()))()((()()())))
    bp = parse_newick('((a,b,(c)),d,((e,f,g)));')
    enmM = rmm(bp.B, bp.B.size)

    # simulating close so only testing open parentheses. A "close" on a closed
    # parenthesis does not make sense, so the result is not useful.
    # In practice, an "close" method should ensure it is operating on a closed
    # parenthesis.
    # [(open_idx, close_idx), ...]
    exp = [(0, 23), (1, 10), (2, 3), (4, 5), (6, 9), (7, 8), (11, 12), 
           (13, 22), (14, 21), (15, 16), (17, 18), (19, 20)]

    for open_, exp_close in exp:
        obs_close = fwdsearch(bp, enmM, open_, -1)
        assert obs_close == exp_close


def test_bwdsearch():
    cdef BP bp
    from bp import parse_newick
    bp = parse_newick('((a,b,(c)),d,((e,f)));')
    enmM = rmm(bp.B, bp.B.size)

    # simulating open so only testing closed parentheses. 
    # [(close_idx, open_idx), ...]
    exp = [(21, 0), (8, 7), (9, 6), (10, 1), (3, 2), (5, 4), (12, 11),
           (16, 15), (20, 13), (19, 14), (18, 17)]

    for close_, exp_open in exp:
        obs_open = bwdsearch(bp, enmM, close_, 0) + 1
        assert obs_open == exp_open

    # slightly modified version of fig2 with an extra child forcing a test
    # of the direct sibling check with negative partial excess

    # this translates into:
    # 012345678901234567890123
    # ((()()(()))()((()()())))
    bp = parse_newick('((a,b,(c)),d,((e,f,g)));')
    enmM = rmm(bp.B, bp.B.size)

    # simulating open so only testing closed parentheses. 
    # [(close_idx, open_idx), ...]
    exp = [(23, 0), (10, 1), (3, 2), (5, 4), (9, 6), (8, 7), (12, 11),
           (22, 13), (21, 14), (16, 15), (18, 17), (20, 19)]

    for close_, exp_open in exp:
        obs_open = bwdsearch(bp, enmM, close_, 0) + 1
        assert obs_open == exp_open


def test_scan_block_forward():
    from bp import parse_newick
    bp = parse_newick('((a,b,(c)),d,((e,f)));')
    
    # [(open, close), ...]
    b = 4
    d = -1
    exp_b_4 = [(0, ((0, -1), (1, -1), (2, 3), (3, -1))),
               (1, ((4, 5), (5, -1), (6, -1), (7, -1))),
                   # 8 and 9 are nonsensical from finding a "close" perspective
               (2, ((8, 9), (9, 10), (10, -1), (11, -1))),  
               (3, ((12, -1), (13, -1), (14, -1), (15, -1))),
                   # 16 and 18 are nonsensical from a "close" perspective
               (4, ((16, 19), (17, 18), (18, 19), (19, -1))),
                   # 20 is nonsensical from finding a "close" perspective
               (5, ((20, 21), (21, -1)))]

    for k, exp_results in exp_b_4:
        for idx, exp_result in exp_results:
            obs_result = scan_block_forward(bp, idx, k, b, bp.excess(idx) + d)
            assert obs_result == exp_result

    b = 8
    exp_b_8 = [(0, ((0, -1), (1, -1), (2, 3), (3, -1), 
                    (4, 5), (5, -1), (6, -1), (7, -1))),
               (1, ((8, 9), (9, 10), (10, -1), (11, 12),
                    (12, -1), (13, -1), (14, -1), (15, -1))),
               (2, ((16, 19), (17, 18), (18, 19), (19, 20), 
                    (20, 21), (21, -1)))]
    
    for k, exp_results in exp_b_8:
        for idx, exp_result in exp_results:
            obs_result = scan_block_forward(bp, idx, k, b, bp.excess(idx) + d)
            assert obs_result == exp_result


def test_scan_block_backward():
    from bp import parse_newick
    bp = parse_newick('((a,b,(c)),d,((e,f)));')
    
    # adding +1 to simluate "open" so calls on open parentheses are weird
    # [(open, close), ...]
    b = 4
    d = 0
    exp_b_4 = [(0, ((0, 0), (1, 0), (2, 0), (3, 2))),
               (1, ((4, 0), (5, 4), (6, 5), (7, 0))),
               (2, ((8, 0), (9, 0), (10, 0), (11, 10))),  
               (3, ((12, 0), (13, 12), (14, 0), (15, 0))),
               (4, ((16, 0), (17, 16), (18, 17), (19, 0))),
               (5, ((20, 0), (21, 0)))]

    for k, exp_results in exp_b_4:
        for idx, exp_result in exp_results:
            obs_result = scan_block_backward(bp, idx, k, b, bp.excess(idx) + d)
            obs_result += 1  # simulating open
            assert obs_result == exp_result

    b = 8
    exp_b_8 = [(0, ((0, 0), (1, 0), (2, 0), (3, 2), 
                    (4, 3), (5, 4), (6, 5), (7, 0))),
               (1, ((8, 0), (9, 0), (10, 0), (11, 10),
                    (12, 11), (13, 12), (14, 9), (15, 8))),
               (2, ((16, 0), (17, 16), (18, 17), (19, 0), 
                    (20, 0), (21, 0)))]                   
    
    for k, exp_results in exp_b_8:
        for idx, exp_result in exp_results:
            obs_result = scan_block_backward(bp, idx, k, b, bp.excess(idx) + d)
            obs_result += 1  # simulating open
            assert obs_result == exp_result


def test_rmm():
    from bp import parse_newick
    # test tree is ((a,b,(c)),d,((e,f)));
    # this is from fig 2 of Cordova and Navarro:
    # http://www.dcc.uchile.cl/~gnavarro/ps/tcs16.2.pdf
    bp = parse_newick('((a,b,(c)),d,((e,f)));')
    exp = np.array([[0, 1, 0, 1, 1, 0, 0, 1, 2, 1, 1, 2, 0],
                    [4, 4, 4, 4, 4, 4, 0, 3, 4, 3, 4, 4, 1]], dtype=np.intp).T
    obs = rmm(bp.B, bp.B.size)

    assert exp.shape[0] == obs.shape[0]
    assert exp.shape[1] == obs.shape[1]
    
    for i in range(exp.shape[0]):
        for j in range(exp.shape[1]):
            assert obs[i, j] == exp[i, j]    
