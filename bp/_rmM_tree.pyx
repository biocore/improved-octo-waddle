# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport ceil, log as ln, pow, log2

import numpy as np
cimport numpy as np

from bp._bp cimport BOOL_t, SIZE_t, BP
from bp._binary_tree cimport * #bt_node_from_left, bt_left_child, bt_right_child
from bp._io import parse_newick  # for test support

np.import_array()

BOOL = np.uint8
SIZE = np.intp

cdef extern from "limits.h":
    int INT_MAX


cdef inline int min(int a, int b) nogil:
    if a > b:
        return b
    else:
        return a


cdef inline int max(int a, int b) nogil:
    if a > b:
        return a
    else:
        return b


cdef np.ndarray construct_rmM_tree(np.ndarray[BOOL_t, ndim=1] B):
    """Construct an rmM tree for improved search over B
    
    Support for section 2 and 3 of Cardova and Navarro

    It should hold that this code works for both a full BP as well as buckets
    a la section 3. 

    THIS CODE IS NOT IN USE. Wasn't able to get the relative aspects working.
    Not sure if that is or is not a loss.
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


cdef SIZE_t[:, :] rmm(BOOL_t[:] B, int B_size) nogil:
    """Construct the rmM tree based off of Navarro and Sadakane

    http://www.dcc.uchile.cl/~gnavarro/ps/talg12.pdf
    """
    cdef int b  # block size
    cdef int n_tip  # number of tips in the binary tree
    cdef int n_internal  # number of internal nodes in the binary tree
    cdef int n_total  # total number of nodes in the binary tree
    cdef int height  # the height of the binary tree
    cdef int i, j, lvl, pos  # for loop support
    cdef int offset  # tip offset in binary tree for a given parenthesis
    cdef int lower_limit  # the lower limit of the bucket a parenthesis is in
    cdef int upper_limit  # the upper limit of the bucket a parenthesis is in
    
    cdef SIZE_t[:, :] mM  # exact min/max per bucket
    cdef int m_idx = 0  # m is minimum excess
    cdef int M_idx = 1  # M is maximum excess
    cdef int min_ = 0 # m, temporary when computing relative
    cdef int max_ = 0 # M, temporary when computing relative
    cdef int excess = 0 # e, temporary when computing relative

    # build tip info
    b = <int>ceil(ln(<double> B_size) * ln(ln(<double> B_size)))

    # determine the number of nodes and height of the binary tree
    n_tip = <int>ceil(B_size / <double> b)
    height = <int>ceil(log2(n_tip))
    n_internal = <int>(pow(2, height)) - 1
    n_total = n_tip + n_internal

    with gil:
        # creation of a memoryview directly or via numpy requires the GIL:
        # http://stackoverflow.com/a/22238012
        mM = np.zeros((n_total, 2), dtype=SIZE)

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
        min_ = INT_MAX
        max_ = 0

        for j in range(lower_limit, upper_limit):
            # G function, a +-1 method where if B[j] == 1 we +1, and if
            # B[j] == 0 we -1
            excess += -1 + (2 * B[j]) 

            if excess < min_:
                min_ = excess

            if excess > max_:
                max_ = excess

            # at the left bound of the bucket
        
        mM[offset + n_internal, m_idx] = min_
        mM[offset + n_internal, M_idx] = max_
        
        i += b

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
            
            if lchild >= n_total:
                continue

            elif rchild >= n_total:
                mM[node, m_idx] = mM[lchild, m_idx] 
                mM[node, M_idx] = mM[lchild, M_idx] 
            else:    
                mM[node, m_idx] = min(mM[lchild, m_idx], mM[rchild, m_idx])
                mM[node, M_idx] = max(mM[lchild, M_idx], mM[rchild, M_idx])

    return mM


cdef inline int scan_block_forward(BP bp, int i, int k, int b, int d) nogil:
    """Scan a block forward from i

    Parameters
    ----------
    bp : BP
        The tree
    i : int
        The index position to start from in the tree
    k : int
        The block to explore
    b : int
        The block size
    d : int
        The depth to search for

    Returns
    -------
    int
        The index position of the result. -1 is returned if a result is not
        found.
    """
    cdef int lower_bound
    cdef int upper_bound
    cdef int j

    # lower_bound is block boundary or right of i
    lower_bound = max(k, 0) * b
    lower_bound = max(i + 1, lower_bound)

    # upper_bound is block boundary or end of tree
    upper_bound = min((k + 1) * b, bp.size)

    for j in range(lower_bound, upper_bound):
        if bp.excess(j) == d:
            return j
    
    return -1


cdef inline int scan_block_backward(BP bp, int i, int k, int b, int d) nogil:
    """Scan a block backward from i

    Parameters
    ----------
    bp : BP
        The tree
    i : int
        The index position to start from in the tree
    k : int
        The block to explore
    b : int
        The block size
    d : int
        The depth to search for

    Returns
    -------
    int
        The index position of the result. -1 is returned if a result is not
        found.
    """
    cdef int lower_bound
    cdef int upper_bound
    cdef int j
    
    # i and k are currently needed to handle the situation where 
    # k_start < i < k_end. It should be possible to resolve using partial 
    # excess.

    # range stop is exclusive, so need to set "stop" at -1 of boundary
    lower_bound = max(k, 0) * b - 1  # is it possible for k to be < 0?
    
    # include the right most position of the k-1 block so we can identify
    # closures spanning blocks. Not positive if this is correct, however if the
    # block is "()((", and we're searching for the opening paired with ")", 
    # we need to go to evaluate the excess prior to the first "(", at least as
    # "open" is defined in Cordova and Navarro
    if lower_bound >= 0:
        lower_bound -= 1
    
    # upper bound is block boundary or left of i, whichever is less
    # note that this is an inclusive boundary since this is a backward search
    upper_bound = min((k + 1) * b, bp.size) - 1
    upper_bound = min(i - 1, upper_bound)
    
    if upper_bound <= 0:
        return -1

    for j in range(upper_bound, lower_bound, -1):
        if bp.excess(j) == d:
            return j

    return -1


cdef int fwdsearch(BP bp, SIZE_t[:, :] mM, int i, int d) nogil:
    """Search forward from i for desired excess

    Parameters
    ----------
    bp : BP
        The tree
    mM : 2D memoryview of SIZE_t
        The binary tree of min and max excess values
    i : int
        The index to search forward from
    d : int
        The excess difference to search for (relative to E[i])
    
    Returns
    -------
    int
        The index of the result, or -1 if no result was found
    """
    cdef int m_idx = 0  # m is minimum excess
    cdef int M_idx = 1  # M is maximum excess
    cdef int b  # the block size
    cdef int n_tip  # the number of tips in the binary tree
    cdef int height  # the height of the binary tree
    cdef int k  # the block being interrogated
    cdef int result  # the result of a scan within a block
    cdef int node  # the node within the binary tree being examined
    
    #TODO: stash binary tree details in a struct/object
    b = <int>ceil(ln(<double> bp.size) * ln(ln(<double> bp.size)))
    n_tip = <int>ceil(bp.size / <double> b)
    height = <int>ceil(log2(n_tip))

    # get the block of parentheses to check
    k = i // b  

    # desired excess
    d += bp.excess(i)

    # see if our result is in our current block
    result = scan_block_forward(bp, i, k, b, d)

    # determine which node our block corresponds too
    node = bt_node_from_left(k, height)
    
    # special case: check sibling
    if result == -1 and bt_is_left_child(node):
        node = bt_right_sibling(node)
        k = node - <int>(pow(2, height) - 1)
        result = scan_block_forward(bp, i, k, b, d)

        # reset node and k in the event that result == -1
        k = i // b
        node = bt_left_sibling(node)
    
    # if we do not have a result, we need to begin traversal of the tree
    if result == -1:
        # walk up the tree
        while not bt_is_root(node):
            if bt_is_left_child(node):
                node = bt_right_sibling(node)
                if mM[node, m_idx] <= d  <= mM[node, M_idx]:
                    break
            node = bt_parent(node)
        
        if not bt_is_root(node):
            # descend until we hit a leaf node
            while not bt_is_leaf(node, height):
                node = bt_left_child(node)

                # evaluate right, if not found, pick left
                if not (mM[node, m_idx] <= d <= mM[node, M_idx]):
                    node = bt_right_sibling(node)
        
        else:
            # no solution
            return -1

        # we have found a block with contains our solution. convert from the
        # node index back into the block index
        k = node - <int>(pow(2, height) - 1)

        # scan for a result using the original d
        result = scan_block_forward(bp, i, k, b, d)

    return result


cdef int bwdsearch(BP bp, SIZE_t[:,:] mM, int i, int d) nogil:
    """Search backward from i for desired excess

    Parameters
    ----------
    bp : BP
        The tree
    mM : 2D memoryview of int
        The binary tree of min and max excess values
    i : int
        The index to search forward from
    d : int
        The excess difference to search for (relative to E[i])
    
    Returns
    -------
    int
        The index of the result, or -1 if no result was found
    """
    cdef int m_idx = 0  # m is minimum excess
    cdef int M_idx = 1  # M is maximum excess
    cdef int b  # the block size
    cdef int n_tip  # the number of tips in the binary tree
    cdef int height  # the height of the binary tree
    cdef int k  # the block being interrogated
    cdef int result  # the result of a scan within a block
    cdef int node  # the node within the binary tree being examined
   
    #TODO: stash details in a struct/object
    b = <int>ceil(ln(<double> bp.size) * ln(ln(<double> bp.size)))
    n_tip = <int>ceil(bp.size / <double> b)
    height = <int>ceil(log2(n_tip))

    # get the block of parentheses to check
    k = i // b  
    
    # desired excess
    d += bp.excess(i)

    # see if our result is in our current block
    result = scan_block_backward(bp, i, k, b, d)

    # determine which node our block corresponds too
    node = bt_node_from_left(k, height)

    # special case: check sibling
    if result == -1 and bt_is_right_child(node):
        node = bt_left_sibling(node)
        k = node - <int>(pow(2, height) - 1)
        result = scan_block_backward(bp, i, k, b, d)
    
        # reset node and k in the event that result == -1
        k = i // b
        node = bt_right_sibling(node)
    
    # if we do not have a result, we need to begin traversal of the tree
    if result == -1:
        while not bt_is_root(node):
            # right nodes cannot contain the solution as we are searching left
            # As such, if we are the right node already, evaluate its sibling.
            if bt_is_right_child(node):
                node = bt_left_sibling(node)
                if mM[node, m_idx] <= d <= mM[node, M_idx]:
                    break
            
            # if we did not find a valid node, adjust for the relative
            # excess of the current node, and ascend to the parent
            node = bt_parent(node)

        # if we did not hit the root, then we have a possible solution
        if not bt_is_root(node):
            # descend until we hit a leaf node
            while not bt_is_leaf(node, height):
                node = bt_right_child(node)

                # evaluate right, if not found, pick left
                if not (mM[node, m_idx] <= d <= mM[node, M_idx]):
                    node = bt_left_sibling(node)

        else:
            # no solution
            return -1

        # we have found a block with contains our solution. convert from the
        # node index back into the block index
        k = node - <int>(pow(2, height) - 1)

        # scan for a result
        result = scan_block_backward(bp, i, k, b, d)
        
    return result


def test_fwdsearch():
    cdef BP bp
    bp = parse_newick('((a,b,(c)),d,((e,f)));')
    enmM = rmm(bp.B, bp.size)

    # simulating close so only testing open parentheses. A "close" on a closed
    # parenthesis does not make sense, so the result is not useful.
    # In practice, an "close" method should ensure it is operating on a closed
    # parenthesis.
    # [(open_idx, close_idx), ...]
    exp = [(1, 10), (0, 21), (2, 3), (4, 5), (6, 9), (7, 8), (11, 12), 
           (13, 20), (14, 19), (15, 16), (17, 18)]

    for open_, exp_close in exp:
        obs_close = fwdsearch(bp, enmM, open_, -1)
        assert obs_close == exp_close

    # slightly modified version of fig2 with an extra child forcing a test
    # of the direct sibling check with negative partial excess

    # this translates into:
    # 012345678901234567890123
    # ((()()(()))()((()()())))
    bp = parse_newick('((a,b,(c)),d,((e,f,g)));')
    enmM = rmm(bp.B, bp.size)

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


def test_bwdsearch():
    cdef BP bp
    bp = parse_newick('((a,b,(c)),d,((e,f)));')
    enmM = rmm(bp.B, bp.size)

    # simulating open so only testing closed parentheses. 
    # [(close_idx, open_idx), ...]
    exp = [(21, 0), (8, 7), (9, 6), (10, 1), (3, 2), (5, 4), (12, 11),
           (16, 15), (20, 13), (19, 14), (18, 17)]

    for close_, exp_open in exp:
        obs_open = bwdsearch(bp, enmM, close_, 0) + 1
        assert obs_open == exp_open

    # slightly modified version of fig2 with an extra child forcing a test
    # of the direct sibling check with negative partial excess

    # this translates into:
    # 012345678901234567890123
    # ((()()(()))()((()()())))
    bp = parse_newick('((a,b,(c)),d,((e,f,g)));')
    enmM = rmm(bp.B, bp.size)

    # simulating open so only testing closed parentheses. 
    # [(close_idx, open_idx), ...]
    exp = [(23, 0), (10, 1), (3, 2), (5, 4), (9, 6), (8, 7), (12, 11),
           (22, 13), (21, 14), (16, 15), (18, 17), (20, 19)]

    for close_, exp_open in exp:
        obs_open = bwdsearch(bp, enmM, close_, 0) + 1
        assert obs_open == exp_open


def test_scan_block_forward():
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
            obs_result = scan_block_forward(bp, idx, k, b, bp.excess(idx) + d)
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
            obs_result = scan_block_forward(bp, idx, k, b, bp.excess(idx) + d)
            assert obs_result == exp_result


def test_scan_block_backward():
    bp = parse_newick('((a,b,(c)),d,((e,f)));')
    
    # adding +1 to simluate "open" so calls on open parentheses are weird
    # [(open, close), ...]
    b = 4
    d = 0
    exp_b_4 = [(0, ((0, 0), (1, 0), (2, 0), (3, 2))),
               (1, ((4, 0), (5, 4), (6, 5), (7, 0))),
               (2, ((8, 0), (9, 0), (10, 0), (11, 10))),  
               (3, ((12, 0), (13, 12), (14, 0), (15, 0))),
               (4, ((16, 0), (17, 16), (18, 17), (19, 0))),
               (5, ((20, 0), (21, 0)))]

    for k, exp_results in exp_b_4:
        for idx, exp_result in exp_results:
            obs_result = scan_block_backward(bp, idx, k, b, bp.excess(idx) + d)
            obs_result += 1  # simulating open
            assert obs_result == exp_result

    b = 8
    exp_b_8 = [(0, ((0, 0), (1, 0), (2, 0), (3, 2), 
                    (4, 3), (5, 4), (6, 5), (7, 0))),
               (1, ((8, 0), (9, 0), (10, 0), (11, 10),
                    (12, 11), (13, 12), (14, 9), (15, 8))),
               (2, ((16, 0), (17, 16), (18, 17), (19, 0), 
                    (20, 0), (21, 0)))]                   
    
    for k, exp_results in exp_b_8:
        for idx, exp_result in exp_results:
            obs_result = scan_block_backward(bp, idx, k, b, bp.excess(idx) + d)
            obs_result += 1  # simulating open
            assert obs_result == exp_result


def test_rmm():
    # test tree is ((a,b,(c)),d,((e,f)));
    # this is from fig 2 of Cordova and Navarro:
    # http://www.dcc.uchile.cl/~gnavarro/ps/tcs16.2.pdf
    bp = parse_newick('((a,b,(c)),d,((e,f)));')
    exp = np.array([[0, 1, 0, 1, 1, 0, 0, 1, 2, 1, 1, 2, 0],
                    [4, 4, 4, 4, 4, 4, 0, 3, 4, 3, 4, 4, 1]], dtype=np.intp).T
    obs = rmm(bp.B, bp.B.size)

    assert exp.shape[0] == obs.shape[0]
    assert exp.shape[1] == obs.shape[1]
    
    for i in range(exp.shape[0]):
        for j in range(exp.shape[1]):
            assert obs[i, j] == exp[i, j]    
