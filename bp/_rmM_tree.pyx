import numpy as np
cimport numpy as np

from bp._bp cimport BOOL_t, SIZE_t
from bp._binary_tree cimport bt_node_from_left, bt_left_child, bt_right_child

np.import_array()

BOOL = np.uint8
SIZE = np.intp

cdef np.ndarray construct_rmM_tree(np.ndarray[BOOL_t, ndim=1] B):

    # build tip info
    b = 4
    n_tip = int(np.ceil(B.size / b))
    height = int(np.ceil(np.log10(n_tip) / np.log10(2)))
    n_internal = int((pow(2, height) - 1))

    print(n_tip, n_internal)
    enmM = np.zeros((n_tip + n_internal, 4), dtype=SIZE)
    # e is total excess
    # n is number of times the minimum occurs
    # m is minimum excess
    # M is maximum excess
    e_idx = 0
    n_idx = 1
    m_idx = 2
    M_idx = 3

    for i in range(0, B.size, b):
        offset = i // b
        lower_limit = i
        upper_limit = min(i + b, B.size)

        min_ = 0 # m
        max_ = 0 # M
        partial_excess = 0 # e
        num_mins = 1 # n
        for j in range(lower_limit, upper_limit):
            if B[j]:
                partial_excess += 1
            else:
                partial_excess -= 1

            if j == lower_limit:
                min_ = partial_excess
                max_ = partial_excess
                num_mins = 1
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

    # see calculations on page 22 of http://www.dcc.uchile.cl/~gnavarro/ps/talg12.pdf
    for lvl in range(height - 1, -1, -1):
        num_curr_nodes = pow(2, lvl)
        for pos in range(num_curr_nodes):
            node = bt_node_from_left(pos, lvl)
            lchild = bt_left_child(node)
            rchild = bt_right_child(node)

            # really just 2 since this is a binary tree...
            print(node, lchild, rchild)
            if lchild >= (n_tip + n_internal):
                continue
            elif rchild >= (n_tip + n_internal):
                enmM[node, e_idx] = enmM[lchild, e_idx]
                enmM[node, m_idx] = min(enmM[lchild, m_idx], enmM[lchild, e_idx])
                enmM[node, M_idx] = max(enmM[lchild, M_idx], enmM[lchild, e_idx])
                enmM[node, n_idx] = enmM[lchild, n_idx]
            else:
                enmM[node, e_idx] = enmM[lchild, e_idx] + enmM[rchild, e_idx]
                enmM[node, m_idx] = min(enmM[lchild, m_idx], 
                                        enmM[lchild, e_idx] + enmM[rchild, m_idx])
                enmM[node, M_idx] = max(enmM[lchild, M_idx], 
                                        enmM[lchild, e_idx] + enmM[rchild, M_idx])

                if enmM[lchild, m_idx] < enmM[lchild, e_idx] + enmM[rchild, m_idx]:
                    enmM[node, n_idx] = enmM[lchild, n_idx]
                elif enmM[lchild, m_idx] > enmM[lchild, e_idx] + enmM[rchild, m_idx]:
                    enmM[node, n_idx] = enmM[rchild, n_idx]
                else:
                    enmM[node, n_idx] = enmM[lchild, n_idx] + enmM[rchild, n_idx]
    return enmM


def test_construct_rmM_tree():
    cdef np.ndarray tree
    cdef np.ndarray obs
    cdef np.ndarray exp

    # test tree is '((a,b,(c)),d,((e,f)));
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
