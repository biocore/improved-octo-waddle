# cython: boundscheck=False, wraparound=False, cdivision=True, linetrace=False
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

import numpy.testing as npt
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


cdef class mM:
    def __cinit__(self, BOOL_t[:] B, int B_size):
        self.m_idx = 0
        self.M_idx = 1
        self.r_idx = 2
        self.k0_idx = 3

        self.rmm(B, B_size)

    cdef void rmm(self, BOOL_t[:] B, int B_size) nogil:
        """Construct the rmM tree based off of Navarro and Sadakane

        http://www.dcc.uchile.cl/~gnavarro/ps/talg12.pdf
        """
        cdef int i, j, lvl, pos  # for loop support
        cdef int offset  # tip offset in binary tree for a given parenthesis
        cdef int lower_limit  # the lower limit of the bucket a parenthesis is in
        cdef int upper_limit  # the upper limit of the bucket a parenthesis is in
        
        cdef SIZE_t[:, :] mM  # exact min/max per bucket
        cdef int min_ = 0 # m, absolute minimum for a blokc
        cdef int max_ = 0 # M, absolute maximum for a block
        #cdef int n = 0
        cdef int excess = 0 # e, absolute excess
        cdef int vbar
        cdef int r = 0
        cdef int k0 = 0

        # build tip info
        self.b = <int>ceil(ln(<double> B_size) * ln(ln(<double> B_size)))

        # determine the number of nodes and height of the binary tree
        self.n_tip = <int>ceil(B_size / <double> self.b)
        self.height = <int>ceil(log2(self.n_tip))
        self.n_internal = <int>(pow(2, self.height)) - 1
        self.n_total = self.n_tip + self.n_internal

        with gil:
            # creation of a memoryview directly or via numpy requires the GIL:
            # http://stackoverflow.com/a/22238012
            self.mM = np.zeros((self.n_total, 4), dtype=SIZE)

        # annoying, cannot do step in range if step is not known at runtime
        # see https://github.com/cython/cython/pull/520
        # for i in range(0, B_size, b):
        # as a result, doing a custom range using a while loop
        # compute for tips of rmM tree
        i = 0
        while i < B_size:
            offset = i // self.b
            lower_limit = i
            upper_limit = min(i + self.b, B_size)
            min_ = INT_MAX
            max_ = 0
            
            self.mM[offset + self.n_internal, self.r_idx] = r
            
            for j in range(lower_limit, upper_limit):
                # G function, a +-1 method where if B[j] == 1 we +1, and if
                # B[j] == 0 we -1
                excess += -1 + (2 * B[j]) 
                r += B[j]
                k0 += 1 - B[j]  # avoid if statement, inc if B[j] == 0

                if excess < min_:
                    min_ = excess

                if excess > max_:
                    max_ = excess

                # at the left bound of the bucket
            
            self.mM[offset + self.n_internal, self.m_idx] = min_
            self.mM[offset + self.n_internal, self.M_idx] = max_
            self.mM[offset + self.n_internal, self.k0_idx] = k0

            i += self.b

        # compute for internal nodes of rmM tree in reverse level order starting 
        # at the level above the tips
        for lvl in range(self.height - 1, -1, -1):
            num_curr_nodes = <int>pow(2, lvl)

            # for each node in the level
            for pos in range(num_curr_nodes):
                # obtain the node, and the index to its children
                node = bt_node_from_left(pos, lvl)
                lchild = bt_left_child(node)
                rchild = bt_right_child(node)
                
                if lchild >= self.n_total:
                    continue

                elif rchild >= self.n_total:
                    self.mM[node, self.m_idx] = self.mM[lchild, self.m_idx] 
                    self.mM[node, self.M_idx] = self.mM[lchild, self.M_idx]
                    self.mM[node, self.k0_idx] = self.mM[lchild, self.k0_idx]
                else:    
                    self.mM[node, self.m_idx] = min(self.mM[lchild, self.m_idx], 
                                                    self.mM[rchild, self.m_idx])
                    self.mM[node, self.M_idx] = max(self.mM[lchild, self.M_idx], 
                                                    self.mM[rchild, self.M_idx])
                    self.mM[node, self.k0_idx] = max(self.mM[lchild, self.k0_idx], 
                                                     self.mM[rchild, self.k0_idx])

                self.mM[node, self.r_idx] = self.mM[lchild, self.r_idx] 
                    

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

#def bench_rank(BP tree):
#    # rank_rmm is ~10x slower, however, it has a 30x reduction in memory
#    # required
#    cdef int i
#    import time
#
#    print("naive r index %d bytes" % (tree._r_index_0.nbytes + 
#                                      tree._r_index_1.nbytes))
#    start = time.time()
#    for i in range(tree.size):
#        tree.rank(0, i)
#        tree.rank(1, i)
#    print("rank(0/1, i): %0.6fs\n" % (time.time() - start))
#
#    print("rmm r data %d bytes" % (tree._rmm.mM.shape[0] * sizeof(SIZE_t)))
#    start = time.time()
#    for i in range(tree.size):
#        tree.rank_rmm(0, i)
#        tree.rank_rmm(1, i)
#    print("rank_rmm(0/1, i): %0.6fs" % (time.time() - start))
#
#    for i in range(tree.size):
#        assert tree.rank_rmm(0, i) == tree.rank(0, i)
#        assert tree.rank_rmm(1, i) == tree.rank(1, i)

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
                  np.ndarray[DOUBLE_t, ndim=1] lengths=None,
                  np.ndarray[object, ndim=1] names=None):
        cdef SIZE_t i
        cdef SIZE_t size
        cdef SIZE_t[:] _e_index
        cdef SIZE_t[:] _k_index_0
        cdef SIZE_t[:] _k_index_1
        cdef SIZE_t[:] _r_index_0
        cdef SIZE_t[:] _r_index_1
        cdef np.ndarray[object, ndim=1] _names
        cdef np.ndarray[DOUBLE_t, ndim=1] _lengths

        # the tree is only valid if it is balanaced
        assert B.sum() == (float(B.size) / 2)
        self.B = B
        self._b_ptr = &B[0]
        self.size = B.size

        self._rmm = mM(B, B.size)

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
        #TODO: leverage rmm tree, and calculate rank on the fly
        _r_index_0 = np.cumsum((1 - B), dtype=SIZE)
        _r_index_1 = np.cumsum(B, dtype=SIZE)

        # construct a select index. These operations are performed frequently,
        # and easy to cache at a relatively minor memory expense. It cannot be
        # assumed that open and close will be same length so can't stack
        #TODO: leverage rmmtree, and calculate select on the fly
        _k_index_0 = np.unique(_r_index_0,
                               return_index=True)[1].astype(SIZE)
        self._k_index_0 = _k_index_0
        _k_index_1 = np.unique(_r_index_1,
                               return_index=True)[1].astype(SIZE)
        self._k_index_1 = _k_index_1

        # construct an excess index. These operations are performed a lot, and
        # similarly can to rank and select, can be cached at a minimal expense.
        #TODO: leverage rmm tree, and calculate excess on the fly
        _e_index = np.empty(B.size, dtype=SIZE)
        for i in range(B.size):
            _e_index[i] = self._excess(i)
        self._e_index = _e_index

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
        return self._lengths[i]

    cpdef inline BPNode get_node(self, SIZE_t i):
        return BPNode(self._names[i], self._lengths[i])

    cdef inline SIZE_t rank(self, SIZE_t t, SIZE_t i) nogil:
        cdef int k
        cdef int r = 0
        cdef int lower_bound
        cdef int upper_bound
        cdef int j
        cdef int node

        #TODO: add method to mM for determining block from i
        k = i // self._rmm.b  
        
        #lower_bound = max(k, 0) * self._rmm.b
        lower_bound = k * self._rmm.b

        # upper_bound is block boundary or end of tree
        upper_bound = min((k + 1) * self._rmm.b, self.size)
        upper_bound = min(upper_bound, i + 1)

        # collect rank from within the block
        for j in range(lower_bound, upper_bound):
            r += self._b_ptr[j]
        
        # collect the rank at the left end of the block
        node = bt_node_from_left(k, self._rmm.height)
        r += self._rmm.mM[node, self._rmm.r_idx]

        # TODO: can this if statement be removed?
        if t:
            return r
        else:
            return (i - r) + 1
         
    #cdef inline SIZE_t rank(self, SIZE_t t, SIZE_t i) nogil:
    #    """The number of occurrences of the bit t in B"""
    #    if t:
    #        return self._r_index_1[i]
    #    else:
    #        return self._r_index_0[i]

    cpdef inline SIZE_t select_rmm(self, SIZE_t t, SIZE_t k):
        node = 0
        if k > self.B.sum():
            return -1

        if t:
            ### i think this mess can be resolved by caching in the rmm whether
            ### a given node has leaves. if no, don't descend

            print('in if')
            # works on test tree so far
            while not bt_is_leaf(node, self._rmm.height):
                print('while loop')
                rchild = bt_right_child(node)
                print(node, bt_left_child(node), rchild, self._rmm.height, self._rmm.n_internal, self._rmm.n_tip, self._rmm.n_total)
                if rchild >= self._rmm.n_total:
                    print("a")
                    lchild = bt_left_child(node)
                    if lchild >= self._rmm.n_total:
                        print("b")
                        if bt_is_left_child(bt_parent(node)):
                            node = bt_right_child(bt_parent(node))
                        else:
                            node = bt_left_sibling(node)
                    else:
                        print("c")
                        node = lchild
                    continue

                if self._rmm.mM[rchild, self._rmm.r_idx] <= k:
                    print('while if loop')
                    # special case: verify there are leaves in this path
                    # TODO: can this be done more efficiently?
                    if not bt_is_leaf(rchild, self._rmm.height) and bt_left_child(rchild) >= self._rmm.n_total:
                        print('while nested if loop')
                        node = bt_left_child(node)
                    else:
                        print('while nested else loop')
                        node = rchild
                else:
                    print('while else loop')
                    node = bt_left_child(node)
            
            lower_bound = (node - self._rmm.n_internal) * self._rmm.b
            upper_bound = min(lower_bound + self._rmm.b, self.size)
            print(node, self._rmm.n_total, lower_bound, upper_bound, self._rmm.b, self._rmm.mM[node, self._rmm.r_idx])
            
            r = -1
            k_ = k - self._rmm.mM[node, self._rmm.r_idx]

            for i in range(lower_bound, upper_bound):
                r += self.B[i]
                if self.B[i] and k_ == r:
                    return i
        else:
            # this all feels wasteful... seems like there should be a cleaner
            # solution here
            while not bt_is_leaf(node, self._rmm.height):
                lchild = bt_left_child(node)

                if k <= self._rmm.mM[lchild, self._rmm.k0_idx]:
                    node = lchild
                else:
                    node = bt_right_child(node)
            
            lower_bound = (node - self._rmm.n_internal) * self._rmm.b
            upper_bound = min(lower_bound + self._rmm.b, self.size)
            
            if bt_is_leaf(bt_left_sibling(node), self._rmm.height):
                k -= self._rmm.mM[bt_left_sibling(node), self._rmm.k0_idx]

            r = 0
            for i in range(lower_bound, upper_bound):
                r += 1 - self.B[i]
                if not self.B[i] and k == r:
                    return i
            
        return -1

    cpdef inline SIZE_t select(self, SIZE_t t, SIZE_t k) nogil:
        """The position in B of the kth occurrence of the bit t."""
        if t:
            return self._k_index_1[k]
        else:
            return self._k_index_0[k]

    cdef SIZE_t _excess(self, SIZE_t i) nogil:
        """Actually compute excess"""
        if i < 0:
            return 0  # wasn't stated as needed but appears so given testing
        return (2 * self.rank(1, i) - i) - 1

    cdef SIZE_t excess(self, SIZE_t i) nogil:
        """the number of opening minus closing parentheses in B[1, i]"""
        # same as: self.rank(1, i) - self.rank(0, i)
        return self._e_index[i]
    
    cdef inline SIZE_t fwdsearch_naive(self, SIZE_t i, int d) nogil:
        """Forward search for excess by depth"""
        cdef:
            SIZE_t j, n = self.size
            SIZE_t b

        b = self._e_index[i] + d

        for j in range(i + 1, n):
            if self._e_index[j] == b:
                return j

        return -1  # wasn't stated as needed but appears so given testing

    cdef inline SIZE_t bwdsearch_naive(self, SIZE_t i, int d) nogil:
        """Backward search for excess by depth"""
        cdef:
            SIZE_t j
            SIZE_t b 

        b = self._e_index[i] + d

        for j in range(i - 1, -1, -1):
            if self._e_index[j] == b:
                return j

        return -1

    cdef inline SIZE_t close(self, SIZE_t i) nogil:
        """The position of the closing parenthesis that matches B[i]"""
        if not self._b_ptr[i]:
            # identity: the close of a closed parenthesis is itself
            return i

        return self.fwdsearch(i, -1)

    cdef inline SIZE_t open(self, SIZE_t i) nogil:
        """The position of the opening parenthesis that matches B[i]"""
        if self._b_ptr[i] or i <= 0:
            # identity: the open of an open parenthesis is itself
            # the open of 0 is open. A negative index cannot be open, so just return
            return i

        return self.bwdsearch(i, 0) + 1

    cdef inline SIZE_t enclose(self, SIZE_t i) nogil:
        """The opening parenthesis of the smallest matching pair that contains position i"""
        if self._b_ptr[i]:
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
        return (BP, (self.B, self._names, self._lengths))

    cdef SIZE_t depth(self, SIZE_t i) nogil:
        """The depth of node i"""
        return self._e_index[i]

    cdef SIZE_t root(self) nogil:
        """The root of the tree"""
        return 0

    cdef SIZE_t parent(self, SIZE_t i) nogil:
        """The parent of node i"""
        return self.enclose(i)

    cdef BOOL_t isleaf(self, SIZE_t i) nogil:
        """Whether the node is a leaf"""
        # publication describe this operation as "iff B[i+1] == 0" which is incorrect

        # most likely there is an implicit conversion here
        return self._b_ptr[i] and (not self._b_ptr[i + 1])

    cdef SIZE_t fchild(self, SIZE_t i) nogil:
        """The first child of i (i.e., the left child)

        fchild(i) = i + 1 (if i is not a leaf)

        Returns 0 if the node is a leaf as the root cannot be a child by
        definition.
        """
        if self._b_ptr[i]:
            if self.isleaf(i):
                return 0
            else:
                return i + 1
        else:
            return self.fchild(self.open(i))

    cdef SIZE_t lchild(self, SIZE_t i) nogil:
        """The last child of i (i.e., the right child)

        lchild(i) = open(close(i) − 1) (if i is not a leaf)

        Returns 0 if the node is a leaf as the root cannot be a child by
        definition.
        """
        if self._b_ptr[i]:
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

    cdef SIZE_t nsibling(self, SIZE_t i) nogil:
        """The next sibling of i (i.e., the sibling to the right)

        nsibling(i) = close(i) + 1 (if the result j holds B[j] = 0 then i has no next sibling)

        Will return 0 if there is no sibling. This makes sense as the root
        cannot have a sibling by definition
        """
        cdef SIZE_t pos

        if self._b_ptr[i]:
            pos = self.close(i) + 1
        else:
            pos = self.nsibling(self.open(i))

        if pos >= self.size:
            return 0
        elif self._b_ptr[pos]:
            return pos
        else:
            return 0

    cdef SIZE_t psibling(self, SIZE_t i) nogil:
        """The previous sibling of i (i.e., the sibling to the left)

        psibling(i) = open(i − 1) (if B[i − 1] = 1 then i has no previous sibling)

        Will return 0 if there is no sibling. This makes sense as the root
        cannot have a sibling by definition
        """
        cdef SIZE_t pos

        if self._b_ptr[i]:
            if self._b_ptr[max(0, i - 1)]:
                return 0

            pos = self.open(i - 1)
        else:
            pos = self.psibling(self.open(i))

        if pos < 0:
            return 0
        elif self._b_ptr[pos]:
            return pos
        else:
            return 0

    def preorder(self, i):
        """Preorder rank of node i"""
        # preorder(i) = rank1(i),
        if self._b_ptr[i]:
            return self.rank(1, i)
        else:
            return self.preorder(self.open(i))

    cpdef SIZE_t preorderselect(self, SIZE_t k) nogil:
        """The node with preorder k"""
        # preorderselect(k) = select1(k),
        return self.select(1, k)

    def postorder(self, i):
        """Postorder rank of node i"""
        # postorder(i) = rank0(close(i)),
        if self._b_ptr[i]:
            return self.rank(0, self.close(i))
        else:
            return self.rank(0, i)

    cpdef SIZE_t postorderselect(self, SIZE_t k) nogil:
        """The node with postorder k"""
        # postorderselect(k) = open(select0(k)),
        return self.open(self.select(0, k))

    def isancestor(self, i, j):
        """Whether i is an ancestor of j"""
        # isancestor(i, j) iff i ≤ j < close(i)
        if i == j:
            return False

        if not self._b_ptr[i]:
            i = self.open(i)

        return i <= j < self.close(i)

    def subtree(self, i):
        """The number of nodes in the subtree of i"""
        # subtree(i) = (close(i) − i + 1)/2.
        if not self._b_ptr[i]:
            i = self.open(i)

        return (self.close(i) - i + 1) / 2

    def levelancestor(self, i, d):
        """ancestor j of i such that depth(j) = depth(i) − d"""
        # levelancestor(i, d) = bwdsearch(i, −d−1)+1
        if d <= 0:
            return -1

        if not self._b_ptr[i]:
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
            SIZE_t i, n = len(tips)
            SIZE_t p, t
            BOOL_t[:] mask
            BOOL_t* mask_ptr

        mask = np.zeros(self.B.size, dtype=BOOL)
        mask_ptr = &mask[0]

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

    cdef BP _mask_from_self(self, BOOL_t[:] mask, 
                            np.ndarray[DOUBLE_t, ndim=1] lengths):
        cdef:
            SIZE_t i, k, n = mask.size, mask_sum = 0
            np.ndarray[BOOL_t, ndim=1] new_b
            np.ndarray[object, ndim=1] new_names
            np.ndarray[object, ndim=1] names = self._names
            np.ndarray[DOUBLE_t, ndim=1] new_lengths
            BOOL_t* new_b_ptr
            BOOL_t* mask_ptr
            DOUBLE_t* lengths_ptr
            DOUBLE_t* new_lengths_ptr

        mask_ptr = &mask[0]
        for i in range(n):
            mask_sum += mask_ptr[i]
        k = 0

        lengths_ptr = &lengths[0]

        new_b = np.empty(mask_sum, dtype=BOOL)
        new_names = np.empty(mask_sum, dtype=object)
        new_lengths = np.empty(mask_sum, dtype=DOUBLE)

        new_b_ptr = &new_b[0]
        new_lengths_ptr = &new_lengths[0]

        for i in range(n):
            if mask_ptr[i]:
                new_b_ptr[k] = self._b_ptr[i]

                # since names is dtype=object, gil is required
                new_names[k] = names[i]
                new_lengths_ptr[k] = lengths_ptr[i]
                k += 1

        return BP(np.asarray(new_b), names=new_names, lengths=new_lengths)

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

        while i < (n - 1):
            if self._b_ptr[i] and not self._b_ptr[i+1]:
                count += 1
                i += 1
            i += 1

        return count

    cdef int scan_block_forward(self, int i, int k, int b, int d) nogil:
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
        upper_bound = min((k + 1) * b, self.size)

        for j in range(lower_bound, upper_bound):
            if self._e_index[j] == d:
                return j
        
        return -1

    cdef int scan_block_backward(self, int i, int k, int b, int d) nogil:
        """Scan a block backward from i

        Parameters
        ----------
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
        upper_bound = min((k + 1) * b, self.size) - 1
        upper_bound = min(i - 1, upper_bound)
        
        if upper_bound <= 0:
            return -1

        for j in range(upper_bound, lower_bound, -1):
            if self.excess(j) == d:
                return j

        return -1

    cdef SIZE_t fwdsearch(self, SIZE_t i, int d) nogil:
        """Search forward from i for desired excess

        Parameters
        ----------
        i : int
            The index to search forward from
        d : int
            The excess difference to search for (relative to E[i])
        
        Returns
        -------
        int
            The index of the result, or -1 if no result was found
        """
        cdef int k  # the block being interrogated
        cdef int result = -1 # the result of a scan within a block
        cdef int node  # the node within the binary tree being examined
        
        # get the block of parentheses to check
        k = i // self._rmm.b  

        # desired excess
        d += self._e_index[i]

        # determine which node our block corresponds too
        node = bt_node_from_left(k, self._rmm.height)
        
        # see if our result is in our current block
        if self._rmm.mM[node, self._rmm.m_idx] <= d <= self._rmm.mM[node, self._rmm.M_idx]:
            result = self.scan_block_forward(i, k, self._rmm.b, d)
        
        # if we do not have a result, we need to begin traversal of the tree
        if result == -1:
            # walk up the tree
            while not bt_is_root(node):
                if bt_is_left_child(node):
                    node = bt_right_sibling(node)
                    if self._rmm.mM[node, self._rmm.m_idx] <= d  <= self._rmm.mM[node, self._rmm.M_idx]:
                        break
                node = bt_parent(node)
            
            if bt_is_root(node):
                return -1

            # descend until we hit a leaf node
            while not bt_is_leaf(node, self._rmm.height):
                node = bt_left_child(node)

                # evaluate right, if not found, pick left
                if not (self._rmm.mM[node, self._rmm.m_idx] <= d <= self._rmm.mM[node, self._rmm.M_idx]):
                    node = bt_right_sibling(node)

            # we have found a block with contains our solution. convert from the
            # node index back into the block index
            k = node - <int>(pow(2, self._rmm.height) - 1)

            # scan for a result using the original d
            result = self.scan_block_forward(i, k, self._rmm.b, d)

        return result

    cdef SIZE_t bwdsearch(self, SIZE_t i, int d) nogil:
        """Search backward from i for desired excess

        Parameters
        ----------
        i : int
            The index to search forward from
        d : int
            The excess difference to search for (relative to E[i])
        
        Returns
        -------
        int
            The index of the result, or -1 if no result was found
        """
        cdef int k  # the block being interrogated
        cdef int result = -1 # the result of a scan within a block
        cdef int node  # the node within the binary tree being examined
      
        # get the block of parentheses to check
        k = i // self._rmm.b  
        
        # desired excess
        d += self.excess(i)

        # see if our result is in our current block
        result = self.scan_block_backward(i, k, self._rmm.b, d)

        # determine which node our block corresponds too
        node = bt_node_from_left(k, self._rmm.height)

        # special case: check sibling
        if result == -1 and bt_is_right_child(node):
            node = bt_left_sibling(node)
            k = node - <int>(pow(2, self._rmm.height) - 1)
            result = self.scan_block_backward(i, k, self._rmm.b, d)
        
           # reset node and k in the event that result == -1
            k = i // self._rmm.b
            node = bt_right_sibling(node)
        
        # if we do not have a result, we need to begin traversal of the tree
        if result == -1:
            while not bt_is_root(node):
                # right nodes cannot contain the solution as we are searching left
                # As such, if we are the right node already, evaluate its sibling.
                if bt_is_right_child(node):
                    node = bt_left_sibling(node)
                    if self._rmm.mM[node, self._rmm.m_idx] <= d <= self._rmm.mM[node, self._rmm.M_idx]:
                        break
                
                # if we did not find a valid node, adjust for the relative
                # excess of the current node, and ascend to the parent
                node = bt_parent(node)
            
            if bt_is_root(node):
                return -1

            # descend until we hit a leaf node
            while not bt_is_leaf(node, self._rmm.height):
                node = bt_right_child(node)

                # evaluate right, if not found, pick left
                if not (self._rmm.mM[node, self._rmm.m_idx] <= d <= self._rmm.mM[node, self._rmm.M_idx]):
                    node = bt_left_sibling(node)

            # we have found a block with contains our solution. convert from the
            # node index back into the block index
            k = node - <int>(pow(2, self._rmm.height) - 1)

            # scan for a result
            result = self.scan_block_backward(i, k, self._rmm.b, d)
            
        return result

    cdef DOUBLE_t unweighted_unifrac(self, SIZE_t[:] u, SIZE_t[:] v) nogil:
        # interesting...
        cdef BOOL_t[:] u_mask
        cdef BOOL_t[:] v_mask
        cdef BOOL_t[:] u_xor_v
        cdef BOOL_t[:] u_or_v
        cdef DOUBLE_t unique = 0.0
        cdef DOUBLE_t total = 0.0
        # can't use & for some reason, angers the gil with _lengths
        cdef DOUBLE_t* lengths_ptr = <DOUBLE_t*>self._lengths.data
        cdef SIZE_t i

        with gil:
            # or just malloc and free
            u_mask = np.zeros(self.size, dtype=BOOL)
            v_mask = np.zeros(self.size, dtype=BOOL)

        u_mask[0] = 1
        u_mask[self.size - 1] = 1

        v_mask[0] = 1
        v_mask[self.size - 1] = 1
        
        for i in range(self.size):
            if u[i]:
                u_mask[i] = 1
                u_mask[self.close(i)] = 1

                p = self.parent(i)
                while p != 0 and u_mask[p] == 0:
                    u_mask[p] = 1
                    u_mask[self.close(p)] = 1

                    p = self.parent(p)
            
            if v[i]:
                v_mask[i] = 1
                v_mask[self.close(i)] = 1

                p = self.parent(i)
                while p != 0 and v_mask[p] == 0:
                    v_mask[p] = 1
                    v_mask[self.close(p)] = 1

                    p = self.parent(p)

        for i in range(self.size):
            if u_mask[i] ^ v_mask[i]:
                unique += lengths_ptr[i]

            if u_mask[i] | v_mask[i]:
                total += lengths_ptr[i]

        if total == 0.0:
            return 0.0
        else:
            return unique / total
###
###
#
# add in .r and .n into rmm calculation
#   - necessary for mincount/minselect
###
###



