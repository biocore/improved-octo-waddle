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
            if e.length is None:
                self.assertEqual(o.length, None if e.is_root() else 0.0)
            else:
                self.assertEqual(o.length, e.length)
            self.assertEqual(o.name, e.name)

        self.assertEqual(obs.ascii_art(), self.sktn.ascii_art())

    def test_from_skbio_treenode(self):
        obs_bp = from_skbio_treenode(self.sktn)
        exp_bp = self.bp

        npt.assert_equal(obs_bp.B, exp_bp.B)
        for i in range(len(self.bp.B)):
            self.assertEqual(exp_bp.name(i), obs_bp.name(i))
            self.assertEqual(exp_bp.length(i), obs_bp.length(i))


if __name__ == '__main__':
    main()
