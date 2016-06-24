from unittest import TestCase, main
from io import StringIO

import skbio
import numpy.testing as npt

from bp import to_skbio_treenode, from_skbio_treenode, parse_newick



class ConversionTests(TestCase):
    def setUp(self):
        self.tstr = "(((a:1,b:2.5)c:6,d:8,(e),(f,g,(h:1,i:2)j:1)k:1.2)l,m:2)r;"
        self.bp = parse_newick(self.tstr)
        self.sktn = skbio.TreeNode.read(StringIO(self.tstr))

    def test_to_skbio_treenode(self):
        obs = to_skbio_treenode(self.bp)
        for o, e in zip(obs.traverse(), self.sktn.traverse()):
            self.assertEqual(o.name, e.name)

            if e.length is None:
                self.assertEqual(o.length, 0.0)
            else:
                self.assertEqual(o.length, e.length)

        self.assertEqual(obs.ascii_art(), self.sktn.ascii_art())

    def test_from_skbio_treenode(self):
        obs_bp, obs_names, obs_lengths = from_skbio_treenode(self.sktn)
        exp_bp, exp_names, exp_lengths = self.bp

        npt.assert_equal(obs_bp.B, exp_bp.B)
        npt.assert_equal(obs_names, exp_names)
        npt.assert_equal(obs_lengths, exp_lengths)


if __name__ == '__main__':
    main()
