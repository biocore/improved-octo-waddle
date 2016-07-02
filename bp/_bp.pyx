#cython: boundscheck=True, wraparound=True
# ----------------------------------------------------------------------------
# Copyright (c) 2013--, BP development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

### NOTE: some doctext strings are copied and pasted from manuscript
### http://www.dcc.uchile.cl/~gnavarro/ps/tcs16.2.pdf

from numpy import float64 as DOUBLE
from numpy import intp as SIZE
from numpy import uint8 as BOOL

import numpy as np
cimport numpy as np
cimport cython
np.import_array()


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
            SIZE_t i
            np.ndarray[SIZE_t, ndim=1] _e_index, _k_index_0, _k_index_1
            np.ndarray[SIZE_t, ndim=1] _closeopen_index
            np.ndarray[object, ndim=1] _names
            np.ndarray[DOUBLE_t, ndim=1] _lengths
            np.ndarray[SIZE_t, ndim=1] _r_index_0, _r_index_1
            SIZE_t* excess_ptr

        # the tree is only valid if it is balanaced
        assert B.sum() == (float(B.size) / 2)
        self.B = B

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

        if names is not None:
            self._names = names
        else:
            self._names = np.full(self.B.size, None, dtype=object)

        if lengths is not None:
            self._lengths = lengths
        else:
            self._lengths = np.zeros(self.B.size, dtype=DOUBLE)

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
        
        n = self.B.size

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

    cpdef inline SIZE_t rank(self, SIZE_t t, SIZE_t i):
        """The number of occurrences of the bit t in B"""
        cdef SIZE_t* _r_index_1_ptr = <SIZE_t*> self._r_index_1.data
        cdef SIZE_t* _r_index_0_ptr = <SIZE_t*> self._r_index_0.data

        if t:
            return _r_index_1_ptr[i]
        else:
            return _r_index_0_ptr[i]

    cpdef inline SIZE_t select(self, SIZE_t t, SIZE_t k):
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

    cpdef inline SIZE_t excess(self, SIZE_t i):
        """the number of opening minus closing parentheses in B[1, i]"""
        # same as: self.rank(1, i) - self.rank(0, i)
        cdef SIZE_t* _e_index_ptr = <SIZE_t*> self._e_index.data
        return _e_index_ptr[i]
    
    cpdef inline SIZE_t fwdsearch(self, SIZE_t i, int d):
        """Forward search for excess by depth"""
        cdef:
            SIZE_t j, n = self.B.size
            SIZE_t b
            SIZE_t* e_index_ptr = <SIZE_t*> self._e_index.data

        b = e_index_ptr[i] + d

        for j in range(i + 1, n):
            if e_index_ptr[j] == b:
                return j

        return -1  # wasn't stated as needed but appears so given testing

    cpdef inline SIZE_t bwdsearch(self, SIZE_t i, int d):
        """Backward search for excess by depth"""
        cdef:
            SIZE_t j
            SIZE_t b ### buffer exception probably from here
            SIZE_t* e_index_ptr = <SIZE_t*> self._e_index.data

        b = e_index_ptr[i] + d

        for j in range(i - 1, -1, -1):
            if e_index_ptr[j] == b:
                return j

        return -1

    cpdef inline SIZE_t close(self, SIZE_t i):
        """The position of the closing parenthesis that matches B[i]"""
        cdef BOOL_t* b_ptr = <BOOL_t*> self.B.data
        cdef SIZE_t* co_ptr = <SIZE_t*> self._closeopen_index.data

        if not b_ptr[i]:
            # identity: the close of a closed parenthesis is itself
            return i

        return co_ptr[i]

    cpdef inline SIZE_t open(self, SIZE_t i):
        """The position of the opening parenthesis that matches B[i]"""
        cdef BOOL_t* b_ptr = <BOOL_t*> self.B.data
        cdef SIZE_t* co_ptr = <SIZE_t*> self._closeopen_index.data

        if b_ptr[i] or i <= 0:
            # identity: the open of an open parenthesis is itself
            # the open of 0 is open. A negative index cannot be open, so just return
            return i

        return co_ptr[i]

    cpdef inline SIZE_t enclose(self, SIZE_t i):
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

    cpdef SIZE_t depth(self, SIZE_t i):
        """The depth of node i"""
        cdef SIZE_t* _e_index_ptr = <SIZE_t*> self._e_index.data
        return _e_index_ptr[i]

    cpdef SIZE_t root(self):
        """The root of the tree"""
        return 0

    cpdef SIZE_t parent(self, SIZE_t i):
        """The parent of node i"""
        return self.enclose(i)

    cpdef inline BOOL_t isleaf(self, SIZE_t i):
        """Whether the node is a leaf"""
        # publication describe this operation as "iff B[i+1] == 0" which is incorrect

        # most likely there is an implicit conversion here
        cdef BOOL_t* B = <BOOL_t*> self.B.data

        return B[i] and (not B[i + 1])

    cpdef SIZE_t fchild(self, SIZE_t i):
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

    cpdef SIZE_t lchild(self, SIZE_t i):
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

    cpdef SIZE_t nsibling(self, SIZE_t i):
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

        if pos >= len(self.B):
            return 0
        elif b_ptr[pos]:
            return pos
        else:
            return 0

    cpdef SIZE_t psibling(self, SIZE_t i):
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

    cpdef inline SIZE_t preorderselect(self, SIZE_t k):
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

    cpdef inline SIZE_t postorderselect(self, SIZE_t k):
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
                if self.name(i) in tips:
                    mask_ptr[i] = 1
                    mask_ptr[i + 1] = 1 # close

                    p = self.parent(i)
                    while p != 0 and mask_ptr[p] == 0:
                        mask_ptr[p] = 1
                        mask_ptr[self.close(p)] = 1

                        p = self.parent(p)

        return self._mask_from_self(mask, self._lengths)

    cdef BP _mask_from_self(self, np.ndarray[BOOL_t, ndim=1] mask, np.ndarray[DOUBLE_t, ndim=1] lengths):
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

        for i in range(self.B.sum()):
            current = self.preorderselect(i)

            if self.isleaf(current):
                mask_ptr[current] = 1
                mask_ptr[self.close(current)] = 1
            else:
                first = self.fchild(current)
                last = self.lchild(current)

                if first == last:
                    new_lengths_ptr[first] = new_lengths_ptr[first] + new_lengths_ptr[current]
                else:
                    mask_ptr[current] = 1
                    mask_ptr[self.close(current)] = 1

        return self._mask_from_self(mask, new_lengths)

    cpdef inline SIZE_t ntips(self):
        cdef:
            SIZE_t i = 0
            SIZE_t count = 0
            SIZE_t n = self.B.size
            BOOL_t* B_ptr

        B_ptr = <BOOL_t*>self.B.data
        while i < (n - 1):
            if B_ptr[i] and not B_ptr[i+1]:
                count += 1
                i += 1
            i += 1

        return count

