# ----------------------------------------------------------------------------
# Copyright (c) 2013--, BP development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
from ._bp import BP
from ._io import parse_newick, write_newick, parse_jplace
from ._conv import to_skbio_treenode, from_skbio_treenode, to_skbio_treearray
from ._insert import insert_fully_resolved


def load(fp, as_treenode=False):
    data = open(fp).read()
    tree = parse_newick(data)
    if as_treenode:
        return to_skbio_treenode(tree)
    else:
        return tree


__all__ = ['BP', 'parse_newick', 'to_skbio_treenode', 'from_skbio_treenode',
           'to_skbio_treearray', 'write_newick', 'parse_jplace',
           'insert_fully_resolved', 'load']

from . import _version
__version__ = _version.get_versions()['version']
