from ._bp import BP

import numpy as np


def _set_node_metadata(ptr, token, names, lengths):
    """Inplace update of names and lengths given token details"""
    name = None
    length = 0.0

    if token.startswith(':'):
        length = float(token[1:])

    elif ':' in token:
        name, length = token.split(':')
        length = float(length)

    else:
        name = token

    names[ptr] = name
    lengths[ptr] = length


def parse_newick(data):
    topology = _newick_to_bp(data)
    open_ = topology.open
    isleaf = topology.isleaf

    names = np.full(len(topology.B), None, dtype=object)
    lengths = np.zeros(len(topology.B), dtype=float)

    ptr = 0
    for token in _tokenizer(data):
        # it may be a performance gain if the tokenizer yielded:
        # (token_type, token). this would allow the elif check to be
        # elif token.type is not structural, resulting in 1 comparison vs
        # 5.

        if token == '(':
            # an open parenthesis never has metadata associated with it
            ptr += 1
            continue

        elif token not in '(),:;':
            _set_node_metadata(open_(ptr), token, names, lengths)
            if isleaf(ptr):
                ptr += 2
            else:
                ptr += 1

    return topology, names, lengths


def _newick_to_bp(data):
    """Convert newick to balanced parentheses

    Newick is _similar_ to BP, but differs notably at the tips of the tree.
    The complexity of the parse below comes from handling tips, and single
    descendents. Examples of situations that introduce this complexity are:

    ((a,b)) -> 11101000
    (a) -> 1100
    () -> 1100
    ((a,b),c) -> 1110100100
    (a,(b,c)) -> 1101101000

    Newick is not required to have node labels on tips, and the interpretation
    of a comma is dependent on prior state.

    The strategy undertaken is to reduce the newick string to only structural
    components. From there, the string is interpreted into tokens of: {"1",
    "0", "10", "100"}, which directly translate into the resulting balanced
    parentheses topology.

    It is very likely the case that this parser can be done better with
    improved efficiency.
    """
    topology = ''.join([c for c in data if c in '(),'])

    parentheses = np.full(len(topology), '', dtype='<U3')

    last = None
    single_descendent = False
    for idx, c in enumerate(topology):
        if c == '(':
            parentheses[idx] = '1'
            single_descendent = True
        elif c == ')':
            if single_descendent or last == ',':
                parentheses[idx] = '100'
                single_descendent = False
            else:
                parentheses[idx] = '0'
        else:
            if last != ')':
                parentheses[idx] = '10'
            single_descendent = False
        last = c

    return BP(np.array([c == '1' for c in ''.join(parentheses)], dtype=bool))


def _tokenizer(data):
    """Yield structural and non-structural slices from data"""
    last = 0
    for idx, i in enumerate(data):
        if i in '(,);':
            if idx - last > 0:
                yield data[last:idx]
            yield i
            last = idx + 1
