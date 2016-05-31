from unittest import TestCase, main
from bp._parse import (parse_newick)
#                       _set_node_metadata)


import numpy as np
import numpy.testing as npt


class NewickTests(TestCase):
    def test_newick_to_topology_simple(self):
        in_ = "();"

        exp = np.array([1, 1, 0, 0], dtype=bool)
        obs = _newick_to_bp(in_).B

        npt.assert_equal(obs, exp)

    def test_newick_to_topology_more(self):
        in_ = "((,),());"

        exp = np.array([1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0], dtype=bool)
        obs = _newick_to_bp(in_).B

        npt.assert_equal(obs, exp)

    def test_newick_to_topology_complex(self):
        in_ = "((a,b),c);"

        exp = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0], dtype=bool)
        obs = _newick_to_bp(in_).B

        npt.assert_equal(obs, exp)

    def test_newick_to_topology_threechildren(self):
        in_ = "((a,b),(c,d),e);"

        exp = np.array([1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0],
                       dtype=bool)
        obs = _newick_to_bp(in_).B

        npt.assert_equal(obs, exp)

    def test_newick_to_topology_nameslengths(self):
        in_ = "((a:2,b):1,(c:4,d)y:20,e);"

        exp = np.array([1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0],
                       dtype=bool)
        obs = _newick_to_bp(in_).B

        npt.assert_equal(obs, exp)

    def test_tokenizer(self):
        in_ = 'asdasd(wex:as:asd)))aa;weq,212)(12)root;'
        exp = ['asdasd', '(', 'wex:as:asd', ')', ')', ')', 'aa;weq', ',',
               '212', ')', '(', '12', ')', 'root', ';']
        obs = list(_tokenizer(in_))
        self.assertEqual(obs, exp)

    def test_set_node_metadata(self):
        n = np.full(5, None, dtype=object)
        l = np.zeros(5)

        exp_n = np.array([None, 'foo', None, 'bar', None], dtype=object)
        exp_l = np.array([0, 0, 1, 2.3, 5])

        _set_node_metadata(1, "foo", n, l)
        _set_node_metadata(2, ":1", n, l)
        _set_node_metadata(3, "bar:2.3", n, l)
        _set_node_metadata(4, ":5", n, l)

        npt.assert_equal(n, exp_n)
        npt.assert_equal(l, exp_l)

    def test_parse_newick(self):
        in_ = "((a:2,b):1,(c:4,d)y:20,e);"

        exp_bp = [1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        exp_n = [None, None, 'a', None, 'b', None, None, 'y', 'c', None, 'd',
                 None, None, 'e', None, None]
        exp_l = [0, 1, 2, 0, 0, 0, 0, 20, 4, 0, 0, 0, 0, 0, 0, 0]

        obs_bp, obs_n, obs_l = parse_newick(in_)

        npt.assert_equal(obs_bp.B, np.asarray(exp_bp, dtype=bool))
        #npt.assert_equal(obs_n, np.asarray(exp_n, dtype=object))
        npt.assert_equal(obs_l, exp_l)

if __name__ == '__main__':
    main()
