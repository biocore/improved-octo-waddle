import skbio
import numpy as np
cimport numpy as np

from ._bp cimport BP


# our noop used when monkey patching invalidate_caches
def noop(*arg, **kwargs):
    pass


def to_skbio_treenode(BP bp):
    """Convert BP to TreeNode

    Parameters
    ----------
    bp : BP
        A BP tree

    Returns
    -------
    skbio.TreeNode
        The tree represented as an skbio.TreeNode
    """
    cdef int i

    nodes = [skbio.TreeNode() for i in range(sum(bp.B))]

    # skbio.TreeNode.append makes a very expensive call to
    # invalidate_caches. Let's remove that from consideration
    # temporarily while constructing the tree
    for node in nodes:
        # monkey patching triggers a weird edge case with python's copy, so the
        # "easy" thing is to disregard what we're doing in copy as these are
        # immutable anyway
        node._exclude_from_copy.add('_old_invalidate_caches')
        node._exclude_from_copy.add('invalidate_caches')
        node._old_invalidate_caches = node.invalidate_caches
        node.invalidate_caches = noop

    root = nodes[0]

    for i in range(sum(bp.B)):
        node_idx = bp.preorderselect(i)
        nodes[i].name = bp.name(node_idx)
        nodes[i].length = bp.length(node_idx)
        nodes[i].edge_num = bp.edge(node_idx)

        if node_idx != bp.root():
            # preorder starts at 1 annoyingly
            parent = bp.preorder(bp.parent(node_idx)) - 1
            nodes[parent].append(nodes[i])

    root.length = None
    
    # ...and then let's restore cache invalidation
    for node in nodes:
        node.invalidate_caches = node._old_invalidate_caches

    return root


def from_skbio_treenode(tree):
    """Convert a skbio TreeNode into BP

    Parameters
    ----------
    tree : skbio.TreeNode
        The tree to convert

    Returns
    -------
    tuple
        (BP, np.array of str, np.array of double)
    """
    n_nodes = len(list(tree.traverse(include_self=True)))

    topo = np.zeros(n_nodes * 2, dtype=np.uint8)
    names = np.full(n_nodes * 2, None, dtype=object)
    lengths = np.zeros(n_nodes * 2, dtype=np.double)
    edges = np.zeros(n_nodes * 2, dtype=np.int32)

    ptr = 0
    seen = set()
    for n in tree.pre_and_postorder(include_self=True):
        if n not in seen:
            topo[ptr] = 1
            names[ptr] = n.name
            lengths[ptr] = n.length or 0.0
            edges[ptr] = getattr(n, 'edge_num', None) or 0

            if n.is_tip():
                ptr += 1

            seen.add(n)

        ptr += 1
    return BP(topo, names=names, lengths=lengths, edges=edges)


def to_skbio_treearray(BP bp):
    """Convert to a tree array comparable to TreeNode.to_array

    Parameters
    ----------
    bp : BP
        A BP tree

    Returns
    -------
    ### TODO: revise
    tuple   ### needs to be a dict keyed by ['length'] and ['child_index']
        np.array of child index positions
        np.array of branch lengths in index order with respect to child index positions
    """
    cdef int i

    class mock_node:
        def __init__(self, id, is_tip):
            self.is_tip_ = is_tip
            self.id = id

        def is_tip(self):
            return self.is_tip_

    child_index = np.zeros((int(bp.B.sum()) - bp.ntips(), 3), dtype=np.int64)
    length = np.zeros(bp.B.sum(), dtype=np.double)
    node_ids = np.zeros(bp.B.size, dtype=np.uint32)
    name = np.full(bp.B.sum(), None, dtype=object)

    # TreeNode.assign_ids, decompose target
    chi_ptr = 0
    cur_index = 0  # the index into node_ids, equivalent to TreeNode.assign_ids
    id_index = dict.fromkeys(set(range(bp.B.sum())))  # map a node's "id" to an object which indicates if it is a leaf or not
    for i in range(bp.B.sum()):
        node_idx = bp.postorderselect(i + 1)  # the index within the BP of the node

        if not bp.isleaf(node_idx):
            fchild = bp.fchild(node_idx)
            lchild = bp.lchild(node_idx)

            sib_idx = fchild  # the sibling index wtihin the BP of the node
            while sib_idx != 0 and sib_idx <= lchild:
                node_ids[sib_idx] = cur_index
                id_index[cur_index] = mock_node(cur_index, bp.isleaf(sib_idx))
                length[cur_index] = bp.length(sib_idx)
                name[cur_index] = bp.name(sib_idx)

                cur_index += 1
                sib_idx = bp.nsibling(sib_idx)

            child_index[chi_ptr] = [node_idx, node_ids[fchild], node_ids[lchild]]
            chi_ptr += 1

    # make sure to capture root
    id_index[bp.B.sum() - 1] = mock_node(cur_index, False)

    node_ids[0] = cur_index
    child_index[:, 0] = node_ids[child_index[:, 0]]
    child_index = child_index[np.argsort(child_index[:, 0])]

    return {'child_index': child_index, 'length': length, 'id_index': id_index,
            'name': name}

