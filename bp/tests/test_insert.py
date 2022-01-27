import unittest
import pkg_resources
from bp import parse_jplace, insert_fully_resolved
import skbio
import pandas as pd


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

    def test_insert_fully_resolved_multiple_placements(self):
        exp = self.final_multiple_fully_resolved
        placements, backbone = parse_jplace(self.jplacedata_multiple)

        # add another placement elsewhere that we would not keep
        # as it's ratio is lower
        dup1 = placements.iloc[0].copy()
        dup1['like_weight_ratio'] -= 0.5
        dup1['edge_num'] += 1

        # add another placement elsewhere that we would not keep
        # as, though its ratio is the same, its pendant is larger
        dup2 = placements.iloc[1].copy()
        dup2['pendant_length'] += 0.5
        dup2['edge_num'] += 1

        placements = pd.concat([placements, pd.DataFrame([dup1, dup2])])

        obs = insert_fully_resolved(placements, backbone)
        self.assertEqual({n.name for n in obs.tips()},
                         {n.name for n in exp.tips()})
        self.assertEqual(obs.compare_rfd(exp), 0)
        self.assertAlmostEqual(obs.compare_tip_distances(exp), 0)

if __name__ == '__main__':
    unittest.main()
