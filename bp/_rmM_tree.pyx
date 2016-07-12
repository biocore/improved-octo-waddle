# cython: cdivision=True
# cython: boundscheck=True
# cython: wraparound=True

from libc.math cimport ceil, log as ln, pow, log2

import numpy as np
cimport numpy as np

from bp._bp cimport BOOL_t, SIZE_t, BP
from bp._binary_tree cimport * #bt_node_from_left, bt_left_child, bt_right_child
from bp._io import parse_newick  # for test support

np.import_array()

BOOL = np.uint8
SIZE = np.intp

# np.ceil as libc.ceil?


cdef np.ndarray construct_rmM_tree(np.ndarray[BOOL_t, ndim=1] B):
    """Construct an rmM tree for improved search over B
    
    Support for section 2 and 3 of Cardova and Navarro

    It should hold that this code works for both a full BP as well as buckets
    a la section 3. 
    """
    cdef int b  # block size
    cdef int n_tip  # number of tips in the binary tree
    cdef int n_internal  # number of internal nodes in the binary tree
    cdef int n_total  # total number of nodes in the binary tree
    cdef int height  # the height of the binary tree
    cdef int B_size = B.size
    cdef int i, j, lvl, pos  # for loop support
    cdef int offset  # tip offset in binary tree for a given parenthesis
    cdef int lower_limit  # the lower limit of the bucket a parenthesis is in
    cdef int upper_limit  # the upper limit of the bucket a parenthesis is in
    
    cdef np.ndarray[SIZE_t, ndim=2] enmM  # relative stats per bucket
    cdef int e_idx = 0  # e is total excess
    cdef int n_idx = 1  # n is number of times the minimum appears
    cdef int m_idx = 2  # m is minimum excess
    cdef int M_idx = 3  # M is maximum excess
    cdef int min_ = 0 # m, temporary when computing relative
    cdef int max_ = 0 # M, temporary when computing relative
    cdef int partial_excess = 0 # e, temporary when computing relative
    cdef int num_mins = 1 # n, temporary when computing relative

    # build tip info
    b = <int>ceil(ln(<double> B_size) * ln(ln(<double> B_size)))

    n_tip = <int>ceil(B_size / <double> b)
    height = <int>ceil(log2(n_tip))
    n_internal = <int>(pow(2, height)) - 1
    n_total = n_tip + n_internal

    enmM = np.zeros((n_total, 4), dtype=SIZE)

    # annoying, cannot do step in range if step is not known at runtime
    # see https://github.com/cython/cython/pull/520
    # for i in range(0, B_size, b):
    # as a result, doing a custom range using a while loop
    # compute for tips of rmM tree
    i = 0
    while i < B_size:
        offset = i // b
        lower_limit = i
        upper_limit = min(i + b, B_size)

        min_ = 0 # m
        max_ = 0 # M
        partial_excess = 0 # e
        num_mins = 1 # n

        for j in range(lower_limit, upper_limit):
            # if we have a parenthesis, the statement below is equivalent too:
            #if B[j]:
            #    partial_excess += 1
            #else:
            #    partial_excess -= 1
            partial_excess += -1 + (2 * B[j])

            # at the left bound of the bucket
            if j == lower_limit:
                min_ = partial_excess
                max_ = partial_excess
                num_mins = 1

            # otherwise update min/max/num 
            else:
                if partial_excess < min_:
                    min_ = partial_excess
                    num_mins = 1
                
                elif partial_excess == min_:
                    num_mins += 1

                if partial_excess > max_:
                    max_ = partial_excess
        
        enmM[offset + n_internal, e_idx] = partial_excess
        enmM[offset + n_internal, m_idx] = min_
        enmM[offset + n_internal, M_idx] = max_
        enmM[offset + n_internal, n_idx] = num_mins
        
        i += b

    # see calculations on page 22 of http://www.dcc.uchile.cl/~gnavarro/ps/talg12.pdf
    # compute for internal nodes of rmM tree in reverse level order starting 
    # at the level above the tips
    for lvl in range(height - 1, -1, -1):
        num_curr_nodes = <int>pow(2, lvl)

        # for each node in the level
        for pos in range(num_curr_nodes):
            # obtain the node, and the index to its children
            node = bt_node_from_left(pos, lvl)
            lchild = bt_left_child(node)
            rchild = bt_right_child(node)

            # if the internal node does not have any children
            if lchild >= n_total:
                continue

            # if the internal node only has a single child, then it must
            # be the left child
            elif rchild >= n_total:
                enmM[node, e_idx] = enmM[lchild, e_idx]
                enmM[node, m_idx] = min(enmM[lchild, m_idx], 
                                        enmM[lchild, e_idx])
                enmM[node, M_idx] = max(enmM[lchild, M_idx], 
                                        enmM[lchild, e_idx])
                enmM[node, n_idx] = enmM[lchild, n_idx]
           
            # otherwise, we have both children
            else:
                enmM[node, e_idx] = enmM[lchild, e_idx] + enmM[rchild, e_idx]
                enmM[node, m_idx] = min(enmM[lchild, m_idx], 
                                        enmM[lchild, e_idx] + \
                                                enmM[rchild, m_idx])
                enmM[node, M_idx] = max(enmM[lchild, M_idx], 
                                        enmM[lchild, e_idx] + \
                                                enmM[rchild, M_idx])

                if enmM[lchild, m_idx] < enmM[lchild, e_idx] + enmM[rchild, m_idx]:
                    enmM[node, n_idx] = enmM[lchild, n_idx]
                elif enmM[lchild, m_idx] > enmM[lchild, e_idx] + enmM[rchild, m_idx]:
                    enmM[node, n_idx] = enmM[rchild, n_idx]
                else:
                    enmM[node, n_idx] = enmM[lchild, n_idx] + \
                            enmM[rchild, n_idx]

    return enmM


def scan_block(bp, i, k, b, d):
    # i and k are currently needed to handle the situation where 
    # k_start < i < k_end. It should be possible to resolve using partial 
    # excess.
    lower_bound = int(max(k, 0) * b)
    lower_bound = max(i, lower_bound)
    upper_bound = int(min((k + 1) * b, bp.B.size))
    for j in range(lower_bound, upper_bound):
        if bp.excess(j) == d:
            return j
    return -1

def test_scan_block():
    bp = parse_newick('((a,b,(c)),d,((e,f)));')
    
    # [(open, close), ...]
    b = 4
    d = -1
    exp_b_4 = [(0, ((0, -1), (1, -1), (2, 3), (3, -1))),
               (1, ((4, 5), (5, -1), (6, -1), (7, -1))),
                   # 8 and 9 are nonsensical from finding a "close" perspective
               (2, ((8, 9), (9, 10), (10, -1), (11, -1))),  
               (3, ((12, -1), (13, -1), (14, -1), (15, -1))),
                   # 16 and 18 are nonsensical from a "close" perspective
               (4, ((16, 19), (17, 18), (18, 19), (19, -1))),
                   # 20 is nonsensical from finding a "close" perspective
               (5, ((20, 21), (21, -1)))]

    for k, exp_results in exp_b_4:
        for idx, exp_result in exp_results:
            obs_result = scan_block(bp, idx, k, b, bp.excess(idx) + d)
            assert obs_result == exp_result

    b = 8
    exp_b_8 = [(0, ((0, -1), (1, -1), (2, 3), (3, -1), 
                    (4, 5), (5, -1), (6, -1), (7, -1))),
               (1, ((8, 9), (9, 10), (10, -1), (11, 12),
                    (12, -1), (13, -1), (14, -1), (15, -1))),
               (2, ((16, 19), (17, 18), (18, 19), (19, 20), 
                    (20, 21), (21, -1)))]
    
    for k, exp_results in exp_b_8:
        for idx, exp_result in exp_results:
            obs_result = scan_block(bp, idx, k, b, bp.excess(idx) + d)
            assert obs_result == exp_result


def fwdsearch(bp, enmM, i, d):
    cdef int e_idx = 0  # e is total excess
    cdef int n_idx = 1  # n is number of times the minimum appears
    cdef int m_idx = 2  # m is minimum excess
    cdef int M_idx = 3  # M is maximum excess
   
    ### could benefit from stashing details in a struct/object
    b = <int>ceil(ln(<double> bp.B.size) * ln(ln(<double> bp.B.size)))
    n_tip = <int>ceil(bp.B.size / <double> b)
    height = <int>ceil(log2(n_tip))
    
    k = i // b  # get the block of parentheses to check
    original_d = d  # retain the original distance for final checking

    # see if our result is in our current block
    result = scan_block(bp, i, k, b, bp.excess(i) + d)

    # determine which node our block corresponds too
    #node = (pow(2, height) - 1) + k
    node = bt_node_from_left(k, height)

    # special case: check sibling
    if result == -1 and bt_is_left_child(node):
        node = bt_right_sibling(node)
        k = node - (pow(2, height) - 1)
        result = scan_block(bp, i, k, b, bp.excess(i) + d)

    # if we do not have a result, we need to begin traversal of the tree
    if result == -1:
        d = d - (max(enmM[node, e_idx], 0) - bp.excess(i))

        # walk up the tree
        while not bt_is_root(node):
            # left nodes cannot contain the solution as the closing
            # parenthesis must be to the right. As such, if we are the
            # left node, evaluate its sibling.
            if bt_is_left_child(node):
                node = bt_right_sibling(node)
                if enmM[node, m_idx] <= d  <= enmM[node, M_idx]:
                    break

            # if we did not find a valid node, adjust for the relative
            # excess of the current node, and ascend to the parent
            d = d - enmM[node, e_idx]
            node = bt_parent(node)

        # if we did not hit the root, then we have a possible solution
        if not bt_is_root(node):
            # descend until we hit a leaf node
            while not bt_is_leaf(node, height):
                node = bt_left_child(node)

                # evaluate left, if not found, pick right
                if not (enmM[node, m_idx] <= d <= enmM[node, M_idx]):
                    node = bt_right_sibling(node)

        else:
            # no solution
            return -1

        # we have found a block with contains our solution. convert from the
        # node index back into the block index
        k = node - (pow(2, height) - 1)

        # scan for a result using the original d
        result = scan_block(bp, i, k, b, bp.excess(i) + original_d)

    return result

def test_construct_rmM_tree():
    cdef np.ndarray tree
    cdef np.ndarray obs
    cdef np.ndarray exp

    # test tree is ((a,b,(c)),d,((e,f)));
    # this is from fig 2 of Cordova and Navarro:
    # http://www.dcc.uchile.cl/~gnavarro/ps/tcs16.2.pdf
    tree = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 
                     0, 0, 0], dtype=np.uint8)
    exp = np.array([[0, 4,-4, 4, 0,-4, 0, 2, 2,-2, 2,-2,-2],
                    [1, 3, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1],
                    [0, 1,-4, 1,-3,-4, 0, 1, 0,-3,-1,-2,-2],
                    [4, 4, 0, 4, 0, 0, 0, 3, 2,-1, 2, 0,-1]], dtype=np.intp).T
    obs = construct_rmM_tree(tree)

    assert exp.shape[0] == obs.shape[0]
    assert exp.shape[1] == obs.shape[1]

    for i in range(exp.shape[0]):
        for j in range(exp.shape[1]):
            assert obs[i, j] == exp[i, j]


def test_fwdsearch():
    cdef BP bp
    # slightly modified version of fig2 with an extra child forcing a test
    # of the direct sibling check with negative partial excess

    # this translates into:
    # 012345678901234567890123
    # ((()()(()))()((()()())))
    bp = parse_newick('((a,b,(c)),d,((e,f,g)));')
    enmM = construct_rmM_tree(bp.B)

    # simulating close so only testing open parentheses. A "close" on a closed
    # parenthesis does not make sense, so the result is not useful.
    # In practice, an "close" method should ensure it is operating on a closed
    # parenthesis.
    # [(open_idx, close_idx), ...]
    exp = [(0, 23), (1, 10), (2, 3), (4, 5), (6, 9), (7, 8), (11, 12), 
           (13, 22), (14, 21), (15, 16), (17, 18), (19, 20)]

    for open_, exp_close in exp:
        obs_close = fwdsearch(bp, enmM, open_, -1)
        assert obs_close == exp_close
