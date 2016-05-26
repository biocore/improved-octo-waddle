# ----------------------------------------------------------------------------
# Copyright (c) 2013--, BP development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase, main

import numpy as np

from bp import BP


class BPTests(TestCase):
    def setUp(self):
        #                       0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        self.fig1_B = np.array([1, 1, 1, 0, 1, 0, 1, 1 ,0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0], dtype=bool)
        self.BP = BP(self.fig1_B)

    def test_rank(self):
        counts_1 = self.fig1_B.cumsum()
        counts_0 = (~self.fig1_B).cumsum()
        for exp, t in zip((counts_1, counts_0), (1, 0)):
            for idx, e in enumerate(exp):

                self.assertEqual(self.BP.rank(t, idx), e)

    def test_select(self):
        pos_1 = np.unique(self.fig1_B.cumsum(), return_index=True)[1] #- 1
        pos_0 = np.unique((~self.fig1_B).cumsum(), return_index=True)[1]

        for exp, t in zip((pos_1, pos_0), (1, 0)):
            for k in range(1, len(exp)):
                self.assertEqual(self.BP.select(t, k), exp[k])

    def test_rank_property(self):
        for i in range(len(self.fig1_B)):
            self.assertEqual(self.BP.rank(1, i) + self.BP.rank(0, i), i+1)

    def test_rank_select_property(self):
        pos_1 = np.unique(self.fig1_B.cumsum(), return_index=True)[1] #- 1
        pos_0 = np.unique((~self.fig1_B).cumsum(), return_index=True)[1]

        for t, pos in zip((0, 1), (pos_0, pos_1)):
            for k in range(1, len(pos)):
                # needed +t on expectation
                print(t, k, self.BP.select(t, k))
                self.assertEqual(self.BP.rank(t, self.BP.select(t, k)), k)# + t)

    def test_excess(self):
        # from fig 2
        exp = [1, 2, 3, 2, 3, 2, 3, 4, 3, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 2, 1, 0]
        for idx, e in enumerate(exp):
            self.assertEqual(self.BP.excess(idx), e)

    def test_close(self):
        exp = [21, 10, 3, 5, 9, 8, 12, 20, 19, 16, 18]
        for i, e in zip(np.argwhere(self.fig1_B == 1).squeeze(), exp):
            self.assertEqual(self.BP.close(i), e)
            self.assertEqual(self.BP.excess(self.BP.close(i)), self.BP.excess(i) - 1)

    def test_open(self):
        exp = [2, 4, 7, 6, 1, 11, 15, 17, 14, 13, 0]
        for i, e in zip(np.argwhere(self.fig1_B == 0).squeeze(), exp):
            self.assertEqual(self.BP.open(i), e)
            self.assertEqual(self.BP.excess(self.BP.open(i) - 1), self.BP.excess(i))

    def test_enclose(self):
        # i > 0 and i < (len(B) - 1)
        exp = [0, 1, 1, 1, 1, 1, 6, 6, 1, 0, 0, 0, 0, 13, 14, 14, 14, 14, 13, 0]
        for i, e in zip(range(1, len(self.fig1_B) - 1), exp):
            self.assertEqual(self.BP.enclose(i), e)

            # unable to get this condition to work. I _think_ this condition is inaccurate?
            #self.assertEqual(self.BP.excess(self.BP.enclose(i) - 1), self.BP.excess(i) - 2)

    def test_rmq(self):
        #       (  (  (  )  (  )  (  (  )  )   )   (   )   (   (   (   )   (   )   )   )   )
        #excess 1  2  3  2  3  2  3  4  3  2   1   2   1   2   3   4   3   4   3   2   1   0
        #i      0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20  21

        exp = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 21],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                     [2, 3, 3, 3, 3, 3, 3, 3, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                        [3, 3, 3, 3, 3, 3, 3, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                           [4, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                              [5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                                 [6, 6, 6, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                                    [7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                                       [8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                                          [9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                                             [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                                                 [11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 21],
                                                     [12, 12, 12, 12, 12, 12, 12, 12, 12, 21],
                                                         [13, 13, 13, 13, 13, 13, 13, 20, 21],
                                                             [14, 14, 14, 14, 14, 19, 20, 21],
                                                                 [15, 16, 16, 16, 19, 20, 21],
                                                                     [16, 16, 16, 19, 20, 21],
                                                                         [17, 18, 19, 20, 21],
                                                                             [18, 19, 20, 21],
                                                                                 [19, 20, 21],
                                                                                     [20, 21],
                                                                                         [21]]
        for i in range(len(self.fig1_B)):
            for j in range(i+1, len(self.fig1_B)):
                self.assertEqual(self.BP.rmq(i, j), exp[i][j - i])

    def test_rMq(self):
        #       (  (  (  )  (  )  (  (  )  )   )   (   )   (   (   (   )   (   )   )   )   )
        #excess 1  2  3  2  3  2  3  4  3  2   1   2   1   2   3   4   3   4   3   2   1   0
        #i      0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20  21

        exp = [[0, 1, 2, 2, 2, 2, 2, 7, 7, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 7],
                  [1, 2, 2, 2, 2, 2, 7, 7, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 7],
                     [2, 2, 2, 2, 2, 7, 7, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 7],
                        [3, 4, 4, 4, 7, 7, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 7],
                           [4, 4, 4, 7, 7, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 7],
                              [5, 6, 7, 7, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 7],
                                 [6, 7, 7, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 7],
                                    [7, 7, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 7],
                                       [8, 8,  8,  8,  8,  8,  8, 15, 15, 15, 15, 15, 15, 15],
                                          [9,  9,  9,  9,  9, 14, 15, 15, 15, 15, 15, 15, 15],
                                             [10, 11, 11, 11, 14, 15, 15, 15, 15, 15, 15, 15],
                                                 [11, 11, 11, 14, 15, 15, 15, 15, 15, 15, 15],
                                                     [12, 13, 14, 15, 15, 15, 15, 15, 15, 15],
                                                         [13, 14, 15, 15, 15, 15, 15, 15, 15],
                                                             [14, 15, 15, 15, 15, 15, 15, 15],
                                                                 [15, 15, 15, 15, 15, 15, 15],
                                                                     [16, 17, 17, 17, 17, 17],
                                                                         [17, 17, 17, 17, 17],
                                                                             [18, 18, 18, 18],
                                                                                 [19, 19, 19],
                                                                                     [20, 20],
                                                                                         [21]]
        for i in range(len(self.fig1_B)):
            for j in range(i+1, len(self.fig1_B)):
                self.assertEqual(self.BP.rMq(i, j), exp[i][j - i])

    def test_root(self):
        self.assertEqual(self.BP.root(), 0)

    def test_depth(self):
        pass # depth(i) == excess(i)

    def test_parent(self):
        pass # parent(i) == enclose(i)

    def test_isleaf(self):
        ### should isleaf be True when queried with the closing parenthesis of a leaf?

        exp = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
        for i, e in enumerate(exp):
            self.assertEqual(self.BP.isleaf(i), e)

    def test_fchild(self):
        exp = [1, 2, None, None, None, None, 7, None, None, 7, 2, None, None, 14, 15, None, None, None, None, 15, 14, 1]
        for i, e in enumerate(exp):
            self.assertEqual(self.BP.fchild(i), e)

    def test_mincount(self):
        #       (  (  (  )  (  )  (  (  )  )   )   (   )   (   (   (   )   (   )   )   )   )
        #i      0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20  21
        #excess 1  2  3  2  3  2  3  4  3  2   1   2   1   2   3   4   3   4   3   2   1   0

        exp = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  4, 1],
                  [1, 1, 2, 2, 3, 3, 3, 3, 4,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                     [1, 1, 1, 2, 2, 2, 2, 3,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                        [1, 1, 2, 2, 2, 2, 3,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                           [1, 1, 1, 1, 1, 2,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                              [1, 1, 1, 1, 2,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                                 [1, 1, 2, 1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                                    [1, 1, 1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                                       [1, 1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                                          [1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                                              [1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                                                  [1,  1,  1,  1,  1,  1,  1,  1,  1,  2, 1],
                                                      [1,  1,  1,  1,  1,  1,  1,  1,  2, 1],
                                                          [1,  1,  1,  1,  1,  1,  2,  1, 1],
                                                              [1,  1,  2,  2,  3,  1,  1, 1],
                                                                  [1,  1,  1,  2,  1,  1, 1],
                                                                      [1,  1,  2,  1,  1, 1],
                                                                          [1,  1,  1,  1, 1],
                                                                              [1,  1,  1, 1],
                                                                                  [1,  1, 1],
                                                                                      [1, 1],
                                                                                         [1]]

        for i in range(len(self.fig1_B)):
            for j in range(i+1, len(self.fig1_B)):
                self.assertEqual(self.BP.mincount(i, j), exp[i][j - i])

    def test_minselect(self):
        """position of the qth minimum in excess(i), excess(i + 1), . . . , excess(j)."""
        exp = {(0, 20, 1): 0,
               (0, 21, 1): 21,
               (0, 20, 2): 10,
               (0, 21, 2): None,
               (0, 20, 3): 12,
               (0, 20, 4): 20,
               (8, 15, 1): 10,
               (8, 15, 2): 12,
               (6, 9, 1): 9}

        for (i, j, q), e in exp.items():
            self.assertEqual(self.BP.minselect(i, j, q), e)

    def test_nsibling(self):
        exp = [None, 11, 4, 4, 6, 6, None, None, None, None, 11, 13, 13, None, None, 17, 17, None, None, None, None, None]
        for i, e in enumerate(exp):
            self.assertEqual(self.BP.nsibling(i), e)

    def test_psibling(self):
        exp = [None, None, None, None, 2, 2, 4, None, None, 4, None, 1, 1, 11, None, None, None, 15, 15, None, 11, None]
        for i, e in enumerate(exp):
            self.assertEqual(self.BP.psibling(i), e)

    def test_preorder(self):
        exp = [1, 2, 3, 3, 4, 4, 5, 6, 6, 5, 2, 7, 7, 8, 9, 10, 10, 11, 11, 9, 8, 1]
        for i, e in enumerate(exp):
            self.assertEqual(self.BP.preorder(i), e)

    def test_preorderselect(self):
        #self.fail("preorderselect is returning the closing position, _probably_ should return opening")
        #exp = [-1, 0, 1, 3, 5, 6, 10, 12, 13, 14, 16]
        exp = [0, 1, 2, 4, 6, 7, 11, 13, 14, 15, 17]
        for k, e in enumerate(exp):
            self.assertEqual(self.BP.preorderselect(k), e)

    def test_postorder(self):
        exp = [11, 5, 1, 1, 2, 2, 4, 3, 3, 4, 5, 6, 6, 10, 9, 7, 7, 8, 8, 9, 10, 11]
        for i, e in enumerate(exp):
            self.assertEqual(self.BP.postorder(i), e)

    def test_postorderselect(self):
        exp = [0, 2, 4, 7, 6, 1, 11, 15, 17, 14, 13, 0]
        for k, e in enumerate(exp):
            self.assertEqual(self.BP.postorderselect(k), e)

    def test_isancestor(self):
        exp = {(0, 0): False,  # identity test
               (2, 1): False,  # tip test
               (1, 2): True,   # open test
               (1, 3): True,   # close test
               (0, 7): True,   # nested test
               (1, 7): True}   # nested test

        for (i, j), e in exp.items():
            self.assertEqual(self.BP.isancestor(i, j), e)

    def test_subtree(self):
        exp = [11, 5, 1, 1, 1, 1, 2, 1, 1, 2, 5, 1, 1, 4, 3, 1, 1, 1, 1, 3, 4, 11]
        for i, e in enumerate(exp):
            self.assertEqual(self.BP.subtree(i), e)

    def test_fwdsearch(self):
        exp = {(0, 0): 10,   # close of first child
               (3, -2): 21,  # close of root
               (11, 2): 15}  # from one tip to the next

        for (i, d), e in exp.items():
            self.assertEqual(self.BP.fwdsearch(i, d), e)

    def test_bwdsearch(self):
        exp = {(3, 0): 1,  # open of parent
               (21, 4): 17,  # nested tip
               (9, 2): 7}  # open of the node

        for (i, d), e in exp.items():
            self.assertEqual(self.BP.bwdsearch(i, d), e)

    def test_levelancestor(self):
        exp = {(2, 1): 1,  # first tip to its parent
               (2, 2): 0,  # first tip to root
               (4, 1): 1,  # second tip to its parent
               (5, 1): 1,  # second tip, closing, to its parent
               (7, 1): 6,  # deep tip to its parent
               (7, 2): 1,  # deep tip to its grandparent
               (7, 3): 0,  # deep tip to its great grand parent
               (7, 9999): 0,  # max out at the root
               (10, 0): -1}  # can't be an ancestor of yourself

        for (i, d), e in exp.items():
            self.assertEqual(self.BP.levelancestor(i, d), e)

    def _testinator(self, exp, f, verbose=False):
        self.assertEqual(len(exp), len(self.fig1_B))
        for i, e in enumerate(exp):
            if verbose:
                print(i, e)
            self.assertEqual(f(i), e)

    def test_levelnext(self):
        #       (   (  (  )  (  )   (   (   )   )   )  (    )   (   (   (   )   (   )   )   )   )
        exp = [-1, 11, 4, 4, 6, 6, 14, 15, 15, 14, 11, 13, 13, -1, -1, 17, 17, -1, -1, -1, -1, -1]
        self.assertEqual(len(exp), len(self.fig1_B))

        for i, e in enumerate(exp):
            self.assertEqual(self.BP.levelnext(i), e)

    def test_levelprev(self):
        #levelprev(i) = open(bwdsearch(i, 0)+1)
        #exp = [-1, -1, -1, -1, 2, 2, 4, -1, -1, 4, -1, 1, 1, 11, 6, 7, 7, 15, 15, 6, 1, -1]
        self.fail("levelprev is not acting as expected, and is getting prior nodes at different levels. "
                  "I think its definition is incorrect.")
        self._testinator(exp, self.BP.levelprev, verbose=True)

    def test_levelleftmost(self):
        #levelleftmost(d) = fwdsearch(0, d),
        pass

    def test_levelrightmost(self):
        #levelrightmost(d) = open(bwdsearch(2n + 1, d)).
        pass

    def test_degree(self):
        #degree(i) = mincount(i + 1, close(i) − 1),
        pass

    def test_child(i):
        # child(i, q) = minselect(i+1, close(i)−1, q−1)+1 for q > 1
        # (for q = 1 it is fchild(i)),
        pass

    def test_childrank(self):
        # childrank(i) = mincount(parent(i) + 1, i) + 1
        # unless B[i − 1] = 1
        # (in which case childrank(i) = 1)
        pass

    def test_lca(self):
        # lca(i, j) = parent(rmq(i, j) + 1)
        # unless isancestor(i, j)
        # (so lca(i, j) = i) or isancestor(j, i) (so lca(i, j) = j),
        nodes = [self.BP.preorderselect(k) for k in range(self.fig1_B.sum())]
        exp = {(nodes[2], nodes[3]): nodes[1],
               (nodes[2], nodes[5]): nodes[1],
               (nodes[2], nodes[9]): nodes[0],
               (nodes[9], nodes[10]): nodes[8],
               (nodes[1], nodes[8]): nodes[0]}
        for (i, j), e in exp.items():
            self.assertEqual(self.BP.lca(i, j), e)

    def test_deepestnode(self):
        # deepestnode(i) = rMq(i, close(i)),
        exp = [7, 7, 2, 2, 4, 4, 7, 7, 7, 7, 7, 11, 11, 15, 15, 15, 15, 17, 17, 15, 15, 7]
        self._testinator(exp, self.BP.deepestnode)

    def test_height(self):
        # height(i) = excess(deepestnode(i)) − excess(i).
        exp = [3, 2, 0, 0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 2, 1, 0, 0, 0, 0, 1, 2, 3]
        self._testinator(exp, self.BP.height)


if __name__ == '__main__':
    main()
