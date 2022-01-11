import unittest
import pkg_resources
from bp import parse_jplace, insert_fully_resolved
import skbio


class InsertTests(unittest.TestCase):
    package = 'bp.tests'
    def setUp(self):
        self.jplacedata_multiple = \
            open(self.get_data_path('300/placement_mul.jplace')).read()
        self.final_multiple_fully_resolved = \
            skbio.TreeNode.read(self.get_data_path('300/placement.full_resolve.newick'))

    def get_data_path(self, filename):
        # adapted from qiime2.plugin.testing.TestPluginBase
        return pkg_resources.resource_filename(self.package,
                                               'data/%s' % filename)

    def test_insert_fully_resolved(self):
        exp = self.final_multiple_fully_resolved
        placements, backbone = parse_jplace(self.jplacedata_multiple)
        obs = insert_fully_resolved(placements, backbone)
        self.assertEqual({n.name for n in obs.tips()},
                         {n.name for n in exp.tips()})
        self.assertEqual(obs.compare_rfd(exp), 0)
        self.assertAlmostEqual(obs.compare_tip_distances(exp), 0)


if __name__ == '__main__':
    unittest.main()
