from unittest import TestCase, main
from bp import parse_newick, to_skbio_treenode

import skbio
import numpy as np
import numpy.testing as npt


class NewickTests(TestCase):
    def test_parse_newick_simple_edge_numbers(self):
        # from https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0031009
        # but without edge labels
        # 0 1 2 3 4 5 6 7 8 9
        # 1 1 1 0 1 0 0 1 0 0
        in_ = '((A:.01{0}, B:.01{1})D:.01{3}, C:.01{4}) {5};'
        exp_sk = '((A:.01, B:.01)D:.01, C:.01);'
        exp = skbio.TreeNode.read([exp_sk])  # skbio doesn't know about edge numbers
        obs = parse_newick(in_)
        obs_sk = to_skbio_treenode(obs)
        self.assertEqual(obs_sk.compare_subsets(exp), 0.0)
        self.assertEqual(obs.edge(2), 0)
        self.assertEqual(obs.edge(4), 1)
        self.assertEqual(obs.edge(1), 3)
        self.assertEqual(obs.edge(7), 4)
        self.assertEqual(obs.edge(0), 5)
        self.assertEqual(obs.edge_from_number(0), 2)
        self.assertEqual(obs.edge_from_number(1), 4)
        self.assertEqual(obs.edge_from_number(3), 1)
        self.assertEqual(obs.edge_from_number(4), 7)
        self.assertEqual(obs.edge_from_number(5), 0)

    def test_parse_newick_nested_quotes(self):
        # bug: quotes are removed
        in_ = '((foo"bar":1,baz:2)x:3)r;'
        exp = skbio.TreeNode.read([in_])
        obs = to_skbio_treenode(parse_newick(in_))
        self.assertEqual(obs.compare_subsets(exp), 0.0)

    def test_parse_newick_with_commas(self):
        # bug: comma is getting interpreted even if in quotes
        in_ = "(('foo,bar':1,baz:2)x:3)r;"
        exp = skbio.TreeNode.read([in_])
        obs = to_skbio_treenode(parse_newick(in_))
        print(obs.ascii_art())
        print(exp.ascii_art())
        self.assertEqual(obs.compare_subsets(exp), 0.0)

    def test_parse_newick_with_parens(self):
        # bug: parens are getting interpreted even if in quotes
        in_ = "(('foo(b)ar':1,baz:2)x:3)r;"
        exp = skbio.TreeNode.read([in_])
        obs = to_skbio_treenode(parse_newick(in_))
        self.assertEqual(obs.compare_subsets(exp), 0.0)

    def test_parse_newick(self):
        in_ = "((a:2,b):1,(c:4,d)y:20,e)r;"

        exp_bp = [1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        exp_n = ['r', None, 'a', None, 'b', None, None, 'y', 'c', None, 'd',
                 None, None, 'e', None, None]
        exp_l = [0, 1, 2, 0, 0, 0, 0, 20, 4, 0, 0, 0, 0, 0, 0, 0]

        obs_bp = parse_newick(in_)

        npt.assert_equal(obs_bp.B, np.asarray(exp_bp, dtype=bool))

        for i, (e_n, e_l) in enumerate(zip(exp_n, exp_l)):
            self.assertEqual(obs_bp.name(i), e_n)
            self.assertEqual(obs_bp.length(i), e_l)

    def test_parse_newick_complex(self):
        in_ = "(((a:1,b:2.5)c:6,d:8,(e),(f,g,(h:1,i:2)j:1)k:1.2)l,m:2)r;"
        #         0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
        exp_bp = [1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1,
                  1, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        exp_n = ['r', 'l', 'c', 'a', None, 'b', None, None, 'd', None, None,
                 'e', None, None, 'k', 'f', None, 'g', None, 'j', 'h', None,
                 'i', None, None, None, None, 'm', None, None]
        exp_l = [0, 0, 6, 1, 0, 2.5, 0, 0, 8, 0, 0, 0, 0, 0, 1.2, 0, 0, 0, 0,
                 1, 1, 0, 2, 0, 0, 0, 0, 2, 0, 0]

        obs_bp = parse_newick(in_)

        npt.assert_equal(obs_bp.B, np.asarray(exp_bp, dtype=bool))

        for i, (e_n, e_l) in enumerate(zip(exp_n, exp_l)):
            self.assertEqual(obs_bp.name(i), e_n)
            self.assertEqual(obs_bp.length(i), e_l)

    def test_parse_newick_singledesc(self):
        in_ = "(((a)b)c,((d)e)f)r;"
        #         0  1  2  3  4  5  6  7  8  9 10 11 12 13
        exp_bp = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        exp_n = ['r', 'c', 'b', 'a', None, None, None, 'f', 'e', 'd', None,
                 None, None, None]

        obs_bp = parse_newick(in_)

        npt.assert_equal(obs_bp.B, np.asarray(exp_bp, dtype=bool))

        for i, e_n in enumerate(exp_n):
            self.assertEqual(obs_bp.name(i), e_n)

    def test_parse_newick_unnamed_singledesc(self):
        in_ = "((a,b)c,d,(e))r;"

        exp_bp = [1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
        exp_n = ['r', 'c', 'a', None, 'b', None, None, 'd', None, None, 'e',
                 None, None, None]

        obs_bp = parse_newick(in_)

        npt.assert_equal(obs_bp.B, np.asarray(exp_bp, dtype=bool))

        for i, e_n in enumerate(exp_n):
            self.assertEqual(obs_bp.name(i), e_n)

    def test_parse_newick_name_with_semicolon(self):
        in_ = "((a,(b,c):5)'d','e; foo':10,((f))g)r;"

        exp_bp = [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0]
        exp_n = ['r', 'd', 'a', None, None, 'b', None, 'c', None, None, None,
                 'e; foo', None, 'g', None, 'f', None, None, None, None]
        exp_l = [0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0]

        obs_bp = parse_newick(in_)

        npt.assert_equal(obs_bp.B, np.asarray(exp_bp, dtype=bool))

        for i, (e_n, e_l) in enumerate(zip(exp_n, exp_l)):
            self.assertEqual(obs_bp.name(i), e_n)
            self.assertEqual(obs_bp.length(i), e_l)


if __name__ == '__main__':
    main()
