# encoding: utf-8
# cython: profile=False, boundscheck=False, wraparound=False

from ._bp cimport BP
import time
import numpy as np
import pandas as pd
import json
cimport numpy as np
cimport cython
np.import_array()


cdef inline np.double_t length_from_edge(unicode token):
    cdef:
        Py_ssize_t split_idx

    # 0.12345{0123} -> 0.12345
    # OR 0.12345[0123] -> 0.12345
    split_idx_curly = token.find('{')
    split_idx_square = token.find('[')
    split_idx = max(split_idx_curly, split_idx_square)
    if split_idx == -1:
        return np.double(token)
    else:
        return np.double(token[:split_idx])


cdef inline np.int32_t number_from_edge(unicode token):
    cdef:
        Py_ssize_t split_idx
        Py_ssize_t end

    # 0.12345{0123} -> 0123
    # OR 0.12345[0123] -> 0123
    split_idx_curly = token.find('{')
    split_idx_square = token.find('[')
    split_idx = max(split_idx_curly, split_idx_square)
    if split_idx == -1:
        return 0
    else:
        end = len(token)
        return np.int32(token[split_idx + 1:end - 1])


cdef void _set_node_metadata(np.uint32_t ptr, unicode token,
                             np.ndarray[object, ndim=1] names, 
                             np.ndarray[np.double_t, ndim=1] lengths,
                             np.ndarray[np.int32_t, ndim=1] edges):
    """Inplace update of names and lengths given token details"""
    cdef:
        np.double_t length
        np.int32_t edge
        Py_ssize_t split_idx, i, end
        unicode name, token_parsed

    name = None
    length = 0.0
    edge = 0

    # NOTE: there is likely some fat to trim in this method. we do a lot
    # of work per token, we could probably do that work smarter. as is,
    # the changes to support edge numbers increase parsing ~20%, which
    # is annoying but probably not a critical
    if token[0] == u':':
        token_parsed = token[1:]
        length = length_from_edge(token_parsed)
        edge = number_from_edge(token_parsed)
    elif u':' in token:
        split_idx = token.rfind(':')
        name = token[:split_idx]
        token_parsed = token[split_idx + 1:]
        length = length_from_edge(token_parsed)
        edge = number_from_edge(token_parsed)
        name = name.strip("'").strip()
    elif u'{' in token or u'[' in token:
        # strip as " {123}" is valid?
        token = token.strip()
        end = len(token)
        edge = np.int32(token.strip()[1:end - 1])
    else:
        name = token.replace("'", "").replace('"', "").strip()

    names[ptr] = name
    lengths[ptr] = length
    edges[ptr] = edge


def write_newick(BP tree, object output, bint include_edge):
    cdef:
        list name_stack
        list edge_stack
        list length_stack
        list open_paren_stack
        object name
        np.npy_float64 length
        Py_ssize_t idx
        np.npy_uint8 v
        Py_ssize_t root_close

    length_stack = []
    name_stack = []
    edge_stack = []
    open_paren_stack = []
    root_close = tree.close(0)

    for idx, v in enumerate(tree.B):
        if v:
            if not tree.isleaf(idx):
                output.write('(')
            name_stack.append(tree.name(idx))
            length_stack.append(tree.length(idx))
            edge_stack.append(tree.edge(idx))
            open_paren_stack.append(idx)
        else:
            name = name_stack.pop()
            length = length_stack.pop()
            edge = edge_stack.pop()

            if name is not None:
                # if we have magical characters, make sure we quote
                if set(name) & {';', ',', '(', ')', ':', '_'}:
                    output.write("'%s'" % name)
                else:
                    output.write(name)

            if include_edge:
                output.write(':%f{%d}' % (length, edge))
            else:
                output.write(':%f' % length)
                
            if tree.nsibling(open_paren_stack.pop()) == 0:
                if idx != root_close:
                    output.write(')')
            else:
                output.write(',')

    output.write(';')


cpdef parse_newick(unicode data):
    cdef:
        np.uint32_t ptr, open_ptr
        Py_ssize_t token_ptr, tmp, lag, datalen
        BP topology
        unicode token, last_token
        np.ndarray[object, ndim=1] names
        np.ndarray[np.double_t, ndim=1] lengths
        np.ndarray[np.int32_t, ndim=1] edges

    if data.count(',') == 0:
        raise ValueError("Only trees with more than 1 node supported")

    data = data.strip()
    if not data.endswith(';'):
        raise ValueError("Newick does not appear terminated with a semicolon")

    datalen = len(data)
    topology = _newick_to_bp(data)

    names = np.full(len(topology.B), None, dtype=object)
    lengths = np.zeros(len(topology.B), dtype=np.double)
    edges = np.full(len(topology.B), 0, dtype=np.int32) 

    ptr = 0
    token_ptr = _ctoken(data, datalen, 0)
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
            # determine if there are unnamed/unlengthed nodes 
            lag += 1

        elif token not in '(),:;':
            ptr += lag
            lag = 0

            open_ptr = topology.open(ptr)
            _set_node_metadata(open_ptr, token, names, lengths, edges)

            if topology.isleaf(ptr):
                ptr += 2
            else:
                ptr += 1

        last_token = token
        tmp = _ctoken(data, datalen, token_ptr)
        token = data[token_ptr:tmp]
        token_ptr = tmp

    topology.set_names(names)
    topology.set_lengths(lengths)
    topology.set_edges(edges)

    return topology


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
        
    potential_single_descendant = False

    topology = np.empty(len(data), dtype=np.uint8)
    topology_ptr = 0
    last_c = u'x'
    in_quote = False

    for i in range(len(data)):
        c = data[i]
        if c == u"'":
            in_quote = not in_quote
        else:
            if in_quote:
                continue
            elif c == u'(':
                # opening of a node
                topology[topology_ptr] = 1
                topology_ptr += 1
                last_c = c
                potential_single_descendant = True
            elif c == u')':
                # closing of a node
                if potential_single_descendant or last_c == u',':
                    # we have a single descendant or a last child (i.e., ",)")
                    topology[topology_ptr] = 1
                    topology[topology_ptr + 1] = 0
                    topology[topology_ptr + 2] = 0
                    topology_ptr += 3
                    potential_single_descendant = False
                else:
                    # it is possible to still have a single descendant in the case
                    # of a multiple single descendant: (...()...)
                    topology[topology_ptr] = 0
                    topology_ptr += 1
                last_c = c
            elif c == u',':
                if last_c != u')':
                    # we have a new tip
                    topology[topology_ptr] = 1
                    topology[topology_ptr + 1] = 0
                    topology_ptr += 2
                potential_single_descendant = False
                last_c = c
            else:
                # ignore non-structure
                pass

    return BP(topology[:topology_ptr])


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


cdef inline int _is_quote(Py_UCS4 c):
    if c == u'"':
        return 1
    elif c == u"'":
        return 1
    else:
        return 0


cdef inline Py_ssize_t _ctoken(unicode data, Py_ssize_t datalen, Py_ssize_t start):
    cdef:
        Py_ssize_t idx, in_quote = 0
        Py_UCS4 c

    if start == (datalen - 1):
        return start + 1

    for idx in range(start, datalen):
        c = data[idx]

        if in_quote:
            if _is_quote(c):
                in_quote = 0
            continue
        else:
            if _is_quote(c):
                in_quote = 1
                continue

        if _ccheck(c):
            if idx == start:
                return idx + 1
            else:
                return idx
    
    return idx + 1


def parse_jplace(object data):
    """Takes a jplace string, returns a DataFrame of placements and the tree

    Implementation specific caveats:

    1) we do not support multiplicities. placements are required to have an "n"
        entry, and we ignore "nm"
    2) Matsen et al (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0031009) 
        define [] for denoting edge labels and {} for denoting edge numbers. We
        currently support either [] OR {}, we do not support edges with both.
        In addition, we REQUIRE the edge labels if specified to be integer.

    If either of these caveats are problems, then we need to modify the code.
    """
    cdef:
        dict as_json
        list fields, placements, fragments, p, placement_data, 
        list placement_inner_data, pquery, entry
        unicode frag, newick
        Py_ssize_t placement_idx, placement_inner_idx, fragment_idx, 
        Py_ssize_t n_fragments
        BP tree

    as_json = json.loads(data)
    newick = as_json['tree']
    placement_data = as_json['placements']

    fields = as_json['fields']
    fields = ['fragment', ] + fields

    placements = []
    for placement_idx in range(len(placement_data)):
        placement = placement_data[placement_idx]
        
        placement_inner_data = placement['p']

        if 'n' not in placement:
            raise KeyError("jplace parsing limited to entries with 'n' keys")

        fragments = placement['n']
        n_fragments = len(fragments)

        for placement_inner_idx in range(len(placement_inner_data)):
            pquery = placement_inner_data[placement_inner_idx]

            for fragment_idx in range(n_fragments):
                frag = fragments[fragment_idx]
                entry = [frag, ] + pquery
                placements.append(entry)

    tree = parse_newick(newick)
    return pd.DataFrame(placements, columns=fields), tree
