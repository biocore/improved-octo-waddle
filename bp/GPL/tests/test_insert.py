import unittest
import pkg_resources
from bp import parse_jplace
from bp.GPL import insert_multifurcating
import skbio


class InsertTests(unittest.TestCase):
    package = 'bp.tests'

    def setUp(self):
        self.jplacedata_multiple = \
            open(self.get_data_path('300/placement_mul.jplace')).read()
        self.final_multiple_multifurcating = \
            skbio.TreeNode.read(self.get_data_path('300/placement_mul.newick'))

    def get_data_path(self, filename):
        # adapted from qiime2.plugin.testing.TestPluginBase
        return pkg_resources.resource_filename(self.package,
                                               '/data/%s' % filename)

    def test_insert_multifurcating(self):
        exp = self.final_multiple_multifurcating
        placements, backbone = parse_jplace(self.jplacedata_multiple)
        obs = insert_multifurcating(placements, backbone)
        self.assertEqual({n.name for n in obs.tips()},
                         {n.name for n in exp.tips()})
        self.assertEqual(obs.compare_rfd(exp), 0)
        self.assertAlmostEqual(obs.compare_tip_distances(exp), 0)


if __name__ == '__main__':
    unittest.main()

