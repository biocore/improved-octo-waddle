# encoding: utf-8
# cython: profile=False, boundscheck=False, wraparound=False

from ._bp cimport BP
from . import to_skbio_treenode
import pandas as pd
import json
import skbio
cimport cython


# see the comment in _insert_setup. Avoid the use of invalidate_caches as it
# is very expensive for tree mutation operations
class TreeNodeNoValidate(skbio.TreeNode):
    def invalidate_caches(self):
        pass


# our noop used when monkey patching invalidate_caches
def noop(*arg, **kwargs):
    pass


# pandas apply functions for preallocation of objects in bulk
def _preallocate_fragment(r):
    return TreeNodeNoValidate(name=r['fragment'], length=r['pendant_length'])


def _preallocate_empty(r):
    return TreeNodeNoValidate()


def _insert_setup(placements, bptree, insert_type):
    # insertion setup addresses:
    # * treenode caching
    # * placement ordering
    # * preallocation of objects where "easy"

    sktree = to_skbio_treenode(bptree)
    node_lookup = {n.edge_num: n for n in sktree.traverse(include_self=True)}

    # mutation operations against TreeNode is expensive as every append or
    # remove triggers a call to invalidate caches, which requires a traversal
    # to find the root (and do other stuff). so let's monkey patch the method
    # to force a noop
    for node in sktree.traverse(include_self=True):
        node.invalidate_caches = noop

    # we are only setup to handle a single placement per fragment, so pull 
    # deduplicated following guidance from Prof. Siavash Mirarab. We sort so
    # "better" has a smaller index value
    # fragment -> group the rows by the fragment, fragment order doesn't matter
    # like_weight_ratio -> our first selection criteria, higher is better
    # pendant_length -> our second selection criteria, lower is better
    placements = placements.sort_values(['fragment', 'like_weight_ratio', 
                                         'pendant_length'],
                                        ascending=[True, False, True])

    # take the first non-duplicated row per fragment. because of the sort, this
    # is assured to be the highest weight ratio, and the smallest pendant 
    # length. Ties are handled arbitrarily. 
    placements = placements[~placements['fragment'].duplicated()]

    if insert_type == 'multifurcating':
        placements = placements.sort_values(['edge_num', 'pendant_length'])
    elif insert_type == 'fully_resolved':
        placements = placements.sort_values(['edge_num', 'distal_length'],
                                            ascending=[True, False])
    else:
        raise ValueError()


    placements['node'] = placements.apply(_preallocate_fragment, axis=1)

    if insert_type == 'fully_resolved':
        placements['parent'] = placements.apply(_preallocate_empty, axis=1)

    return placements, sktree, node_lookup


# pd.DataFrame is not a resolved type so we cannot use it here for cython
def insert_fully_resolved(object placements, BP bptree):
    """Update the backbone, fully resolving edges with multiple queries

    Parameters
    ----------
    placements : pd.DataFrame
        jplace data represented as a DataFrame
    bptree : bp.BP
        An instance of a BP tree, this is expected to contain edge numbers
        and correspond to the backbone for the jplace data

    Returns
    -------
    skbio.TreeNode
        A tree with the fragments placed
    """
    # TODO: profile, type and re-profile
    placements, sktree, node_lookup = \
        _insert_setup(placements, bptree, 'fully_resolved')

    for edge, edge_grp in placements.groupby('edge_num'):
        existing_node = node_lookup[edge]
        current_parent = existing_node.parent
        
        # break the edge
        current_parent.remove(existing_node)
        existing_node.parent = None
        existing_length = existing_node.length

        for _, fragment in edge_grp.iterrows():
            distal_length = fragment['distal_length']
            fragment_node = fragment['node']
            fragment_parent = fragment['parent']

            # update branch lengths
            fragment_parent.length = existing_length - distal_length
            existing_length = distal_length

            # attach the nodes
            fragment_parent.append(fragment_node)
            current_parent.append(fragment_parent)

            # update
            current_parent = fragment_parent
        
        existing_node.length = existing_length
        current_parent.append(existing_node)
        existing_node.length = distal_length

    return sktree
