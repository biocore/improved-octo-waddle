# encoding: utf-8
# cython: profile=False, boundscheck=False, wraparound=False

from .._bp cimport BP
from bp._insert import _insert_setup, TreeNodeNoValidate
from .. import to_skbio_treenode
import pandas as pd
import json
import skbio
cimport cython


# this method described here was derived from Genesis
# https://github.com/lczech/genesis
# The license for Genesis is included herein under
# Genesis_LICENSE.txt
def insert_multifurcating(object placements, BP bptree):
    """Update the backbone, with multifurcation for edges with multiple queries

    Parameters
    ----------
    placements : pd.DataFrame
        jplace data represented as a DataFrame
    bptree : bp.BP
        An instance of a BP tree, this is expected to contain edge numbers
        and correspond to the backbone for the jplace data

    Note
    ----
    This method was derived directly from the Genesis codebase, and is
    therefore GPL.

    Returns
    -------
    skbio.TreeNode
        A tree with the fragments placed
    """
    # TODO: profile, type and re-profile
    placements, sktree, node_lookup = \
        _insert_setup(placements, bptree, 'multifurcating')

    # it is much faster to bulk allocate than construct on the fly, so let's
    # do that
    new_parents = [TreeNodeNoValidate()
                   for _ in range(len(placements['edge_num'].unique()))]
    parent_idx = 0

    new_bridges = [TreeNodeNoValidate()
                   for _ in range(placements['edge_num'].duplicated().sum())]
    bridge_idx = 0

    for edge, edge_grp in placements.groupby('edge_num'):
        # update topology
        existing_node = node_lookup[edge]
        current_parent = existing_node.parent
        current_parent.remove(existing_node)

        # get a pre-allocated node
        new_parent = new_parents[parent_idx]
        parent_idx += 1  # make sure we update our offset

        # gather detail on our minimal placement
        min_frag = edge_grp.iloc[0]
        min_pendant = min_frag['pendant_length']
        min_distal = min_frag['distal_length']
        new_node = min_frag['node']

        if len(edge_grp) > 1:
            # if we have multiple placements, we construct a node that contains
            # all of the placements, and place this under a "bridge" such that
            # it is a sister to the existing node.

            # derived from
            # https://github.com/lczech/genesis/blob/98c064d8e3e2efaa97da33c9263f6fef3724f0a5/lib/genesis/placement/function/tree.cpp#L295

            # update topology
            bridge = new_bridges[bridge_idx]
            bridge_idx += 1
            bridge.append(existing_node)
            bridge.append(new_parent)
            current_parent.append(bridge)
            new_parent.append(new_node)

            # Gappa uses the average distal for joining the node encompassing
            # placements and existing back to the tree. As we are subtracting
            # against the existing node, it is possible to introduce a negative
            # branch length, so we enforce a minimum of 0.0
            avg_prox_len = edge_grp['distal_length'].mean()
            bridge.length = max(0.0, existing_node.length - avg_prox_len)

            # update edges. the first node we place has a length of zero as
            # its parent accounts for its pendant
            existing_node.length = avg_prox_len
            new_parent.length = min_pendant
            new_node.length = 0.0

            for i in range(1, len(edge_grp)):
                # gather placement detail
                frag_row = edge_grp.iloc[i]
                frag_node = frag_row['node']

                # update topology
                new_parent.append(frag_node)

                # update the branch length. Note that "frag_node" has its
                # length first set to its pendant during the preallocation
                # step. we subtract the parent pendant to avoid counting
                # extra edge length. This should never be < 0 as the data
                # are presorted by pendant length, so the fragment evaluated
                # here is assured to have a > length than the pendant used
                # with the parent.
                frag_node.length = frag_node.length - new_parent.length
        else:
            # if we only have a single placement, we place the fragment as a
            # sister to the existing node.

            # update topology
            current_parent.append(new_parent)
            new_parent.append(existing_node)
            new_parent.append(new_node)

            # update branch lengths
            new_node.length = min_pendant
            new_parent.length = existing_node.length - min_distal
            existing_node.length = min_distal

    return sktree
