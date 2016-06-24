# encoding: utf-8
# cython: profile=True

from ._bp cimport BP
import time
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _set_node_metadata(np.uint32_t ptr, unicode token, 
                             np.ndarray[object, ndim=1] names, 
                             np.ndarray[np.double_t, ndim=1] lengths):
    """Inplace update of names and lengths given token details"""
    cdef:
        np.double_t length
        Py_ssize_t split_idx, i
        unicode name

    name = None
    length = 0.0

    if token[0] == u':':
        length = np.double(token[1:])
    elif ':' in token:
        split_idx = token.rfind(':')
        name = token[:split_idx]
        length = np.double(token[split_idx + 1:])
        name = name.replace("'", "").replace('"', "")
    else:
        name = token.replace("'", "").replace('"', "")

    names[ptr] = name
    lengths[ptr] = length


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef parse_newick(unicode data):
    cdef:
        np.uint32_t ptr, open_ptr
        Py_ssize_t token_ptr, tmp, lag
        object topology, open_, isleaf
        unicode token, last_token
        np.ndarray[object, ndim=1] names
        np.ndarray[np.double_t, ndim=1] lengths

    ##### probably can cache tip indices
    ##### probably can cache parent indices

    topology = _newick_to_bp(data)

    names = np.full(len(topology.B), None, dtype=object)
    lengths = np.zeros(len(topology.B), dtype=np.double)

    ptr = 0
    token_ptr = _ctoken(data, 0)
    token = data[0:token_ptr]
    last_token = None

    # lag reflects the scenario where ((x))y, where the label y gets may end
    # up being associated with an earlier unnamed vertex. lag represents the
    # offset between the topology pointer and the token pointer effectively.
    lag = 0
    while token != ';':
        if token == '(':
            # an open parenthesis never has metadata associated with it
            ptr += 1
        
        if (token == ')' or token == ',') and last_token == ')':
            lag += 1

        elif token not in '(),:;':
            ptr += lag
            lag = 0

            open_ptr = topology.open(ptr)
            _set_node_metadata(open_ptr, token, names, lengths)

            if topology.isleaf(ptr):
                ptr += 2
            else:
                ptr += 1

        last_token = token
        tmp = _ctoken(data, token_ptr)
        token = data[token_ptr:tmp]
        token_ptr = tmp
    
    return topology, names, lengths


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _newick_to_bp(unicode data):
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
    cdef:
        Py_ssize_t i, topology_ptr, single_descendent
        Py_UCS4 c, last_c
        np.ndarray[np.uint8_t, ndim=1] topology
        
    single_descendent = False

    topology = np.empty(len(data), dtype=np.uint8)
    topology_ptr = 0
    last_c = u'x'

    for i in range(len(data)):
        # ignore non-structure

        c = data[i]
        if c == u'(':
            topology[topology_ptr] = 1
            topology_ptr += 1
            last_c = c
            single_descendent = True
        elif c == u')':
            if single_descendent or last_c == u',':
                topology[topology_ptr] = 1
                topology[topology_ptr + 1] = 0
                topology[topology_ptr + 2] = 0
                topology_ptr += 3
                single_descendent = False
            else:
                topology[topology_ptr] = 0
                topology_ptr += 1
            last_c = c
        elif c == u',':
            if last_c != u')':
                topology[topology_ptr] = 1
                topology[topology_ptr + 1] = 0
                topology_ptr += 2
            single_descendent = False
            last_c = c
         
    return BP(topology[:topology_ptr])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int _ccheck(Py_UCS4 c):
    """structure check"""
    cdef:
        Py_ssize_t i

    if c == u'(':
        return 1
    elif c == u')':
        return 1
    elif c == u',':
        return 1
    elif c == u';':
        return 1
    else:
        return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Py_ssize_t _ctoken(unicode data, Py_ssize_t start):
    cdef:
        Py_ssize_t idx, datalen
        Py_UCS4 c

    datalen = len(data)

    if start == (datalen - 1):
        return start + 1

    for idx in range(start, datalen):
        c = data[idx]

        if _ccheck(c):
            if idx == start:
                return idx + 1
            else:
                return idx
    
    return idx + 1
