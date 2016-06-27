# ----------------------------------------------------------------------------
# Copyright (c) 2013--, BP development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

### NOTE: some doctext strings are copied and pasted from manuscript
### http://www.dcc.uchile.cl/~gnavarro/ps/tcs16.2.pdf
import numpy as np
cimport numpy as np
cimport cython


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
    def __cinit__(self, unicode name, np.double_t length):
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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, np.ndarray[np.uint8_t, ndim=1] B,
                  np.ndarray[np.uint32_t, ndim=1] closeopen=None,
                  np.ndarray[object, ndim=1] names=None,
                  np.ndarray[np.double_t, ndim=1] lengths=None):
        cdef:
            Py_ssize_t i
            np.ndarray[np.uint32_t, ndim=1] _e_index, _k_index_0, _k_index_1
            np.ndarray[np.uint32_t, ndim=1] _closeopen_index
            np.ndarray[object, ndim=1] _names
            np.ndarray[np.double_t, ndim=1] _lengths
            np.ndarray[np.uint32_t, ndim=2] _r_index

        # the tree is only valid if it is balanaced
        assert B.sum() == (float(B.size) / 2)
        self.B = B

        # construct a rank index. These operations are performed frequently,
        # and easy to cache at a relatively minor memory expense
        _r_index = np.vstack([np.cumsum((1 - B), dtype=np.uint32), 
                              np.cumsum(B, dtype=np.uint32)])
        self._r_index = _r_index

        # construct a select index. These operations are performed frequently,
        # and easy to cache at a relatively minor memory expense. It cannot be
        # assumed that open and close will be same length so can't stack
        _k_index_0 = np.unique(_r_index[0], 
                               return_index=True)[1].astype(np.uint32)
        self._k_index_0 = _k_index_0
        _k_index_1 = np.unique(_r_index[1], 
                               return_index=True)[1].astype(np.uint32)
        self._k_index_1 = _k_index_1

        # construct an excess index. These operations are performed a lot, and
        # similarly can to rank and select, can be cached at a minimal expense.
        _e_index = np.empty(B.size, dtype=np.uint32)
        for i in range(B.size):
            _e_index[i] = self._excess(i)
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
            self._lengths = np.zeros(self.B.size, dtype=np.double)

    def set_names(self, np.ndarray[object, ndim=1] names):
        self._names = names

    def set_lengths(self, np.ndarray[np.double_t, ndim=1] names):
        self._lengths = names

    cpdef inline unicode name(self, Py_ssize_t i):
        return self._names[i]

    cpdef inline np.double_t length(self, Py_ssize_t i):
        return self._lengths[i]

    cpdef inline BPNode get_node(self, Py_ssize_t i):
        return BPNode(self._names[i], self._lengths[i])

    cdef inline void _set_closeopen_cache(self):
        cdef:
            Py_ssize_t i, j, n, m
            np.ndarray[np.uint32_t, ndim=1] closeopen
            np.ndarray[np.uint8_t, ndim=1] b
        
        b = self.B
        n = b.size
        closeopen = np.zeros(n, dtype=np.uint32)

        for i in range(n):
            # if we haven't already cached it (cheaper than open/close call)
            # note: idx 0 is valid, but it will be on idx -1 and correspond to
            # root so this is safe.
            if closeopen[i] == 0:
                if b[i]:
                    j = self.close(i)
                else:
                    j = self.open(i)

                closeopen[i] = j
                closeopen[j] = i
        
        self._closeopen_index = closeopen

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef inline np.uint32_t rank(self, Py_ssize_t t, Py_ssize_t i):
        """The number of occurrences of the bit t in B"""
        return self._r_index[t, i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef inline np.uint32_t select(self, Py_ssize_t t, Py_ssize_t k):
        """The position in B of the kth occurrence of the bit t."""
        if t:
            return self._k_index_1[k]
        else:
            return self._k_index_0[k]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline np.uint32_t _excess(self, Py_ssize_t i):
        """Actually compute excess"""
        if i < 0:
            return 0  # wasn't stated as needed but appears so given testing
        return (2 * self.rank(1, i) - i) - 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef inline np.uint32_t excess(self, Py_ssize_t i):
        """the number of opening minus closing parentheses in B[1, i]"""
        # same as: self.rank(1, i) - self.rank(0, i)
        return self._e_index[i]
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef inline np.int32_t fwdsearch(self, Py_ssize_t i, int d):
        """Forward search for excess by depth"""
        cdef:
            int j, n = self.B.size
            np.int32_t b ### buffer exception probably from here
            np.ndarray[np.uint32_t, ndim=1] e_index

        e_index = self._e_index
        b = e_index[i] + d

        for j in range(i + 1, n):
            if e_index[j] == b:
                return j

        return -1  # wasn't stated as needed but appears so given testing

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef inline Py_ssize_t bwdsearch(self, Py_ssize_t i, int d):
        """Backward search for excess by depth"""
        cdef:
            int j
            np.int32_t b ### buffer exception probably from here
            np.ndarray[np.uint32_t, ndim=1] e_index
    
        e_index = self._e_index
        b = e_index[i] + d

        for j in range(i - 1, -1, -1):
            if e_index[j] == b:
                return j

        return -1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef inline np.int32_t close(self, Py_ssize_t i):
        """The position of the closing parenthesis that matches B[i]"""
        cdef:
            np.ndarray[np.uint32_t, ndim=1] coi = self._closeopen_index

        if not self.B[i]:
            # identity: the close of a closed parenthesis is itself
            return i

        if coi is not None:
            return coi[i]
        else:
            return self.fwdsearch(i, -1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef inline np.int32_t open(self, Py_ssize_t i):
        """The position of the opening parenthesis that matches B[i]"""
        cdef:
            np.ndarray[np.uint32_t, ndim=1] coi = self._closeopen_index

        if self.B[i] or i <= 0:
            # identity: the open of an open parenthesis is itself
            # the open of 0 is open. A negative index cannot be open, so just return
            return i
        ### if bwdsearch returns -1, should check and dump None?
        if coi is not None:
            return coi[i]
        else:
            return self.bwdsearch(i, 0) + 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef inline np.int32_t enclose(self, Py_ssize_t i):
        """The opening parenthesis of the smallest matching pair that contains position i"""
        if self.B[i]:
            return self.bwdsearch(i, -2) + 1
        else:
            return self.bwdsearch(i - 1, -2) + 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.uint32_t rmq(self, Py_ssize_t i, Py_ssize_t j):
        """The leftmost minimum excess in i -> j"""
        cdef:
            Py_ssize_t k, min_k
            np.uint32_t min_v, obs_v

        min_k = i
        min_v = self.excess(i)  # a value larger than what will be tested
        for k in range(i, j + 1):
            obs_v = self.excess(k)
            if obs_v < min_v:
                min_k = k
                min_v = obs_v
        return min_k

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.uint32_t rMq(self, Py_ssize_t i, Py_ssize_t j):
        """The leftmost maximmum excess in i -> j"""
        cdef:
            Py_ssize_t k, max_k
            np.uint32_t max_v, obs_v

        max_k = i
        max_v = self.excess(i)  # a value larger than what will be tested
        for k in range(i, j + 1):
            obs_v = self.excess(k)
            if obs_v > max_v:
                max_k = k
                max_v = obs_v

        return max_k

        # return np.array([self.excess(k) for k in range(i, j + 1)]).argmax() + i

    def depth(self, i):
        """The depth of node i"""
        return self.excess(i)

    def root(self):
        """The root of the tree"""
        return 0

    def parent(self, i):
        """The parent of node i"""
        return self.enclose(i)

    cpdef inline np.uint8_t isleaf(self, Py_ssize_t i):
        """Whether the node is a leaf"""
        # publication describe this operation as "iff B[i+1] == 0" which is incorrect

        # most likely there is an implicit conversion here
        return self.B[i] and (not self.B[i + 1])

    def fchild(self, i):
        """The first child of i (i.e., the left child)

        fchild(i) = i + 1 (if i is not a leaf)"""
        if self.B[i]:
            if not self.isleaf(i):
                return i + 1
        else:
            return self.fchild(self.open(i))

    def lchild(self, i):
        """The last child of i (i.e., the right child)

        lchild(i) = open(close(i) − 1) (if i is not a leaf)"""
        if self.B[i]:
            if not self.isleaf(i):
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

    def nsibling(self, i):
        """The next sibling of i (i.e., the sibling to the right)

        nsibling(i) = close(i) + 1 (if the result j holds B[j] = 0 then i has no next sibling)"""
        if self.B[i]:
            pos = self.close(i) + 1
        else:
            pos = self.nsibling(self.open(i))

        if pos is None:
            return None
        elif pos >= len(self.B):
            return None
        elif self.B[pos]:
            return pos
        else:
            return None

    def psibling(self, i):
        """The previous sibling of i (i.e., the sibling to the left)

        psibling(i) = open(i − 1) (if B[i − 1] = 1 then i has no previous sibling)"""
        if self.B[i]:
            if self.B[max(0, i - 1)]:
                return None
            pos = self.open(i - 1)
        else:
            pos = self.psibling(self.open(i))

        if pos is None:
            return None
        elif pos < 0:
            return None
        elif self.B[pos]:
            return pos
        else:
            return None

    def preorder(self, i):
        """Preorder rank of node i"""
        # preorder(i) = rank1(i),
        if self.B[i]:
            return self.rank(1, i)
        else:
            return self.preorder(self.open(i))


    def preorderselect(self, k):
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

    cpdef inline np.uint32_t postorderselect(self, Py_ssize_t k):
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

    cpdef BP shear(self, np.ndarray[np.uint32_t, ndim=1] tips):
        cdef:
            Py_ssize_t i, n = tips.size
            np.uint32_t p, t
            np.ndarray[np.uint8_t, ndim=1] mask

        mask = np.zeros(self.B.size, dtype=np.uint8)
        mask[self.root()] = 1
        mask[self.close(self.root())] = 1

        for i in range(n):
            t = tips[i]

            mask[t] = 1
            mask[t + 1] = 1 # close

            p = self.parent(t)
            while mask[p] == 0 and p != 0:
                mask[p] = 1
                mask[self.close(p)] = 1

                p = self.parent(p)

        return self._mask_from_self(mask, self._lengths)

    cdef BP _mask_from_self(self, np.ndarray[np.uint8_t, ndim=1] mask, np.ndarray[np.double_t, ndim=1] lengths):
        cdef:
            Py_ssize_t i, k, n = mask.size, mask_sum = mask.sum()
            np.ndarray[np.uint8_t, ndim=1] b, new_b
            np.ndarray[object, ndim=1] names, new_names
            np.ndarray[np.double_t, ndim=1] new_lengths

        k = 0
        b = self.B
        names = self._names

        new_b = np.empty(mask_sum, dtype=np.uint8)
        new_names = np.empty(mask_sum, dtype=object)
        new_lengths = np.empty(mask_sum, dtype=np.double)

        for i in range(n):
            if mask[i]:
                new_b[k] = b[i]
                new_names[k] = names[i]
                new_lengths[k] = lengths[i]
                k += 1

        return BP(new_b, names=new_names, lengths=new_lengths)

    cpdef BP collapse(self):
        cdef:
            Py_ssize_t i, n = self.B.sum()
            np.uint32_t current, first, last
            np.ndarray[np.uint8_t, ndim=1] collapse_mask

        mask = np.zeros(self.B.size, dtype=np.uint8)
        mask[self.root()] = 1
        mask[self.close(self.root())] = 1

        new_lengths = self._lengths.copy()

        for i in range(self.B.sum()):
            current = self.preorderselect(i)

            if self.isleaf(current):
                mask[current] = 1
                mask[self.close(current)] = 1
            else:
                first = self.fchild(current)
                last = self.lchild(current)

                if first == last:
                    new_lengths[first] = new_lengths[first] + new_lengths[current]
                else:
                    mask[current] = 1
                    mask[self.close(current)] = 1

        return self._mask_from_self(mask, new_lengths)

    cpdef inline np.uint32_t ntips(self):
        cdef:
            Py_ssize_t i = 0
            np.uint32_t count = 0

        while i < (self.B.size - 1):
            if self.B[i] and not self.B[i+1]:
                count += 1
                i += 1
            i += 1

        return count

