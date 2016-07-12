# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# An implementation of a complete binary tree in breadth first order adapted 
# from https://github.com/jfuentess/sea2015/blob/master/binary_trees.h

from libc.math cimport pow
from bp._bp cimport SIZE_t


cdef inline SIZE_t bt_is_root(SIZE_t v) nogil:
    """Is v the root"""
    return v == 0


cdef inline SIZE_t bt_is_left_child(SIZE_t v) nogil:
    """Is v a left child of some node"""
    return 0 if bt_is_root(v) else v % 2


cdef inline SIZE_t bt_is_right_child(SIZE_t v) nogil:
    """Is v a right child of some node"""
    return 0 if bt_is_root(v) else 1 - (v % 2)


cdef inline SIZE_t bt_parent(SIZE_t v) nogil:
    """Get the index of the parent of v"""
    return 0 if bt_is_root(v) else (v - 1) // 2


cdef inline SIZE_t bt_left_child(SIZE_t v) nogil:
    """Get the index of the left child of v"""
    return 2 * v + 1


cdef inline SIZE_t bt_right_child(SIZE_t v) nogil:
    """Get the index of the right child of v"""
    return 2 * v + 2


cdef inline SIZE_t bt_left_sibling(SIZE_t v) nogil:
    """Get the index of the left sibling of v"""
    return v - 1


cdef inline SIZE_t bt_right_sibling(SIZE_t v) nogil:
    """Get the index of the right sibling of v"""
    return v + 1


cdef inline SIZE_t bt_is_leaf(SIZE_t v, SIZE_t height) nogil:
    """Determine if v is a leaf"""
    return (v >= pow(2, height) - 1)


cdef inline SIZE_t bt_node_from_left(SIZE_t pos, SIZE_t height) nogil:
    """Get the index from the left of a node at a given height"""
    return <SIZE_t>(pow(2, height)) - 1 + pos


