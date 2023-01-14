import sys

import cloudpickle
import numpy as np

from pymc_bart.tree import Node, get_depth


def test_split_node():
    split_node = Node().new_split_node(5, 3.0, 2)
    assert split_node.index == 5
    assert get_depth(split_node.index) == 2
    assert split_node.value == 3.0
    assert split_node.idx_split_variable == 2
    assert len(split_node.idx_data_points) == 0
    assert split_node.get_idx_parent_node() == 2
    assert split_node.get_idx_left_child() == 11
    assert split_node.get_idx_right_child() == 12
    assert split_node.is_split_node() is True
    assert split_node.is_leaf_node() is False

    #cloudpickle.dumps(split_node)
    print("Memory usage of the object:", sys.getsizeof(split_node), "bytes")

    ## https://github.com/pymc-devs/pymc/blob/6ab0c034fa4bb7df30f11ba1fa92a3f47f987bb3/pymc/sampling/parallel.py#L401
    ## f: https://github.com/numba/numba/issues/1846


def test_leaf_node():
    leaf_node = Node().new_leaf_node(5, 3.14, np.ones(3, np.int32))
    assert leaf_node.index == 5
    assert get_depth(leaf_node.index) == 2
    assert leaf_node.value == 3.14
    assert leaf_node.idx_split_variable == -1
    assert np.array_equal(leaf_node.idx_data_points, [1, 1, 1])
    assert leaf_node.get_idx_parent_node() == 2
    assert leaf_node.get_idx_left_child() == 11
    assert leaf_node.get_idx_right_child() == 12
    assert leaf_node.is_split_node() is False
    assert leaf_node.is_leaf_node() is True
