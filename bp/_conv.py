import skbio
import numpy as np

from ._bp import BP


def to_skbio_treenode(bp):
    """Convert BP to TreeNode

    Parameters
    ----------
    bp : tuple
        Tuple of (BP, np.array of str, np.array of double)

    Returns
    -------
    skbio.TreeNode
        The tree represented as an skbio.TreeNode
    """
    topo, names, lengths = bp
    nodes = [skbio.TreeNode() for i in range(sum(topo.B))]
    root = nodes[0]

    for i in range(sum(topo.B)):
        node_idx = topo.preorderselect(i)
        nodes[i].name = names[node_idx]
        nodes[i].length = lengths[node_idx]

        if node_idx != topo.root():
            # preorder starts at 1 annoyingly
            parent = topo.preorder(topo.parent(node_idx)) - 1
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
    return (BP(topo), names, lengths)
