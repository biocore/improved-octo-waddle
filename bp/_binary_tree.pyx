def test_binary_tree():
    # wikipedia example, https://en.wikipedia.org/wiki/Binary_tree#Arrays
    
    # root test
    assert bt_is_root(0)
    assert bt_left_child(0) == 1
    assert bt_right_child(0) == 2
    assert bt_node_from_left(0, 0) == 0
    assert bt_offset_from_left(0) == 0 
    assert bt_offset_from_right(0) == 0 
    assert bt_left_leaf(0, 3) == 7
    assert bt_right_leaf(0, 3) == 14

    # lvl 1
    assert bt_is_left_child(1)
    assert bt_is_right_child(2)
    assert bt_left_child(1) == 3
    assert bt_right_child(1) == 4
    assert bt_left_child(2) == 5
    assert bt_right_child(2) == 6
    assert bt_parent(1) == 0
    assert bt_parent(2) == 0
    assert bt_right_sibling(1) == 2
    assert bt_left_sibling(2) == 1
    assert bt_node_from_left(0, 1) == 1
    assert bt_node_from_left(1, 1) == 2
    assert bt_offset_from_left(1) == 0 
    assert bt_offset_from_left(2) == 1 
    assert bt_offset_from_right(1) == 1 
    assert bt_offset_from_right(2) == 0 
    assert bt_left_leaf(1, 3) == 7
    assert bt_left_leaf(2, 3) == 11
    assert bt_right_leaf(1, 3) == 10
    assert bt_right_leaf(2, 3) == 14
    
    # lvl 2
    assert bt_is_left_child(3)
    assert bt_is_right_child(4)
    assert bt_is_left_child(5)
    assert bt_is_right_child(6)
    assert bt_parent(3) == 1
    assert bt_parent(4) == 1
    assert bt_parent(5) == 2
    assert bt_parent(6) == 2
    assert bt_right_sibling(3) == 4
    assert bt_left_sibling(4) == 3
    assert bt_right_sibling(5) == 6
    assert bt_left_sibling(6) == 5
    assert bt_node_from_left(0, 2) == 3
    assert bt_node_from_left(1, 2) == 4
    assert bt_node_from_left(2, 2) == 5
    assert bt_node_from_left(3, 2) == 6
    assert bt_offset_from_left(3) == 0 
    assert bt_offset_from_left(4) == 1 
    assert bt_offset_from_left(5) == 2 
    assert bt_offset_from_left(6) == 3
    assert bt_offset_from_right(3) == 3
    assert bt_offset_from_right(4) == 2 
    assert bt_offset_from_right(5) == 1 
    assert bt_offset_from_right(6) == 0
    assert bt_left_leaf(3, 3) == 7
    assert bt_left_leaf(4, 3) == 9
    assert bt_left_leaf(5, 3) == 11
    assert bt_left_leaf(6, 3) == 13
    assert bt_right_leaf(3, 3) == 8
    assert bt_right_leaf(4, 3) == 10
    assert bt_right_leaf(5, 3) == 12
    assert bt_right_leaf(6, 3) == 14

    # lvl 3
    assert bt_offset_from_left(7) == 0
    assert bt_offset_from_left(8) == 1
    assert bt_offset_from_left(9) == 2
    assert bt_offset_from_left(10) == 3
    assert bt_offset_from_left(11) == 4
    assert bt_offset_from_left(12) == 5
    assert bt_offset_from_left(13) == 6
    assert bt_offset_from_left(14) == 7
    assert bt_offset_from_right(7) == 7
    assert bt_offset_from_right(8) == 6
    assert bt_offset_from_right(9) == 5
    assert bt_offset_from_right(10) == 4
    assert bt_offset_from_right(11) == 3
    assert bt_offset_from_right(12) == 2
    assert bt_offset_from_right(13) == 1
    assert bt_offset_from_right(14) == 0
    
