import skbio
import numpy as np

from ._bp import BP


def to_skbio_treenode(bp):
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
    nodes = [skbio.TreeNode() for i in range(sum(bp.B))]
    root = nodes[0]

    for i in range(sum(bp.B)):
        node_idx = bp.preorderselect(i)
        nodes[i].name = bp.name(node_idx)
        nodes[i].length = bp.length(node_idx)

        if node_idx != bp.root():
            # preorder starts at 1 annoyingly
            parent = bp.preorder(bp.parent(node_idx)) - 1
            nodes[parent].append(nodes[i])

    root.length = None
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

    ptr = 0
    seen = set()
    for n in tree.pre_and_postorder(include_self=True):
        if n not in seen:
            topo[ptr] = 1
            names[ptr] = n.name
            lengths[ptr] = n.length or 0.0

            if n.is_tip():
                ptr += 1

            seen.add(n)

        ptr += 1
    return BP(topo, names=names, lengths=lengths)


def to_skbio_treearray(bp):
    """Convert to a tree array comparable to TreeNode.to_array

    Parameters
    ----------
    bp : BP
        A BP tree

    Returns
    -------
    tuple   ### needs to be a dict keyed by ['length'] and ['child_index']
        np.array of child index positions
        np.array of branch lengths in index order with respect to child index positions
    """
    class mock_node:
        def __init__(self, is_tip):
            self._is_tip = is_tip
        def is_tip(self):
            return self._is_tip

    child_index = np.zeros((int(bp.B.sum()) - bp.ntips(), 3), dtype=np.uint32)
    length = np.zeros(bp.B.sum(), dtype=np.double)
    node_ids = np.zeros(bp.B.size, dtype=np.uint32)

    # TreeNode.assign_ids, decompose target
    cur_index = 0  # the index into node_ids
    id_index = {}  # map a node's "id" to an object which indicates if it is a leaf or not
    for i in range(bp.B.sum()):
        node_idx = bp.postorderselect(i + 1)  # the index within the BP of the node
        if not bp.isleaf(node_idx):
            id_index[i] = mock_node(False)
            fchild = bp.fchild(node_idx)
            lchild = bp.lchild(node_idx)

            sib_idx = fchild  # the sibling index wtihin the BP of the node
            while sib_idx is not None and sib_idx <= lchild:
                node_ids[sib_idx] = cur_index
                id_index[cur_index] = mock_node(bp.isleaf(sib_idx))
                cur_index += 1
                sib_idx = bp.nsibling(sib_idx)

    node_ids[0] = cur_index

    # TreeNode.to_array, note, not capturing names and using a "nan_length_value" of 0.0
    ch_ptr = 0
    for i in range(bp.B.sum()):
        node_idx = bp.postorderselect(i + 1)  # preorderselect does not need +1. Possible side effect due to caching

        length[node_ids[node_idx]] = bp.length(node_idx)
        if not bp.isleaf(node_idx):
            child_index[ch_ptr] = [node_ids[node_idx], node_ids[bp.fchild(node_idx)], node_ids[bp.lchild(node_idx)]]
            ch_ptr += 1

    return {'child_index': child_index, 'length': length, 'id_index': id_index}

