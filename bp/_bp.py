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

"""root the tree root
levelleftmost(d) / levelrightmost(d) leftmost/rightmost node with depth d
lca(i, j) the lowest common ancestor of two nodes i, j
deepestnode(i) the (first) deepest node in the subtree of i
height(i) the height of i (distance to its deepest node)
degree(i) q = number of children of node i
child(i, q) q-th child of node i
childrank(i) q = number of siblings to the left of node i
leafrank(i) number of leaves to the left and up to node i
leafselect(k) kth leaf of the tree
numleaves(i) number of leaves in the subtree of node i
leftmostleaf(i) / rightmostleaf(i) leftmost/rightmost leaf of node i"""

class BP(object):
    def __init__(self, B):
        assert B.sum() == (float(B.size) / 2)

        self.B = B

        self._k_index = [np.unique((~self.B).cumsum(), return_index=True)[1],
                         np.unique(self.B.cumsum(), return_index=True)[1] ]#- 1]
        self._r_index = [(~self.B).cumsum(), self.B.cumsum()]

    def rank(self, t, i):
        """The number of occurrences of the bit t in B"""
        return self._r_index[t][i]

    def select(self, t, k):
        """The position in B of the kth occurrence of the bit t."""
        return self._k_index[t][k]

    def excess(self, i):
        """the number of opening minus closing parentheses in B[1, i]"""
        # same as: self.rank(1, i) - self.rank(0, i)
        if i < 0:
            return 0  # wasn't stated as needed but appears so given testing
        return (2 * self.rank(1, i) - i) - 1

    def fwdsearch(self, i, d):
        """Forward search for excess by depth"""
        # cache excess, need to bench on large trees as it may be slower as it avoids shortcut
        # (self._e[i+1:] == (self._e[i] + d)).get_first_true_value()

        # but, definite cython target
        for j in range(i + 1, len(self.B)):
            if self.excess(j) == (self.excess(i) + d):
                return j
        return -1 # wasn't stated as needed but appears so given testing

    def bwdsearch(self, i, d):
        """Backward search for excess by depth"""
        # cache excess
        # (self._e[:i] == (self._e[i] + d)).get_last_true_value()
        for j in range(0, i)[::-1]:
            if self.excess(j) == (self.excess(i) + d):
                return j
        return -1 # wasn't stated as needed but appears so given testing

    def close(self, i):
        """The position of the closing parenthesis that matches B[i]"""
        if not self.B[i]:
            # identity: the close of a closed parenthesis is itself
            return i

        return self.fwdsearch(i, -1)

    def open(self, i):
        """The position of the opening parenthesis that matches B[i]"""
        if self.B[i]:
            # identity: the open of an open parenthesis is itself
            return i

        if i <= 0:
            # the open of 0 is open. A negative index cannot be open, so just return
            return i

        ### if bwdsearch returns -1, should check and dump None?
        return self.bwdsearch(i, 0) + 1

    def enclose(self, i):
        """The opening parenthesis of the smallest matching pair that contains position i"""
        if self.B[i]:
            return self.bwdsearch(i, -2) + 1
        else:
            return self.bwdsearch(i - 1, -2) + 1

    def rmq(self, i, j):
        """The leftmost minimum excess in i -> j"""
        return np.array([self.excess(k) for k in range(i, j + 1)]).argmin() + i

    def rMq(self, i, j):
        """The leftmost maximmum excess in i -> j"""
        return np.array([self.excess(k) for k in range(i, j + 1)]).argmax() + i

    def depth(self, i):
        """The depth of node i"""
        return self.excess(i)

    def root(self):
        """The root of the tree"""
        return 0

    def parent(self, i):
        """The parent of node i"""
        return self.enclose(i)

    def isleaf(self, i):
        """Whether the node is a leaf"""
        # publication describe this operation as "iff B[i+1] == 0" which is incorrect
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
        pass

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

    def postorderselect(self, k):
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
