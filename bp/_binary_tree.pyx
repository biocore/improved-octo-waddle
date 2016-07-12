def test_binary_tree():
    # wikipedia example, https://en.wikipedia.org/wiki/Binary_tree#Arrays
    
    # root test
    assert bt_is_root(0)
    assert bt_left_child(0) == 1
    assert bt_right_child(0) == 2
    assert bt_node_from_left(0, 0) == 0

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
