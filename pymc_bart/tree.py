#   Copyright 2022 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import math

from copy import deepcopy
from functools import lru_cache

from numba import int64, float64, typeof, types, njit, int32
from numba.experimental import jitclass
from pytensor import config
import numpy as np


@jitclass(spec=[
    ("index", int64),
    ("value", float64),
    ("idx_split_variable", int64),
    ("idx_data_points", types.Array(dtype=int32, ndim=1, layout="A"),)
])
class Node:
    def __init__(self):
        self.index = -1
        self.value = 0.0
        self.idx_data_points = np.empty(0, np.int32)
        self.idx_split_variable = -1

    def new_leaf_node(self, index, value, idx_data_points):
        self.index = index
        self.value = value
        self.idx_data_points = idx_data_points
        self.idx_split_variable = -1

        return self

    def new_split_node(self, index, split_value, idx_split_variable):
        self.index = index
        self.value = split_value
        self.idx_data_points = np.empty(0, np.int32)
        self.idx_split_variable = idx_split_variable

        return self

    def __getstate__(self):
        return (self.index, self.value, self.idx_data_points, self.idx_split_variable)

    def __setstate__(self, state):
        self.index, self.value, self.idx_data_points, self.idx_split_variable = state

    def __reduce__(self):
        return (Node, (self.index, self.value, self.idx_data_points, self.idx_split_variable))

    def get_idx_parent_node(self) -> int:
        return (self.index - 1) // 2

    def get_idx_left_child(self) -> int:
        return self.index * 2 + 1

    def get_idx_right_child(self) -> int:
        return self.get_idx_left_child() + 1

    def is_split_node(self) -> bool:
        return self.idx_split_variable >= 0

    def is_leaf_node(self) -> bool:
        return not self.is_split_node()


@lru_cache
def get_depth(index: int) -> int:
    return int(math.floor(math.log(index + 1, 2)))


@jitclass(spec=[
    ("tree_structure", types.DictType(int64, Node.class_type.instance_type)),
    ("idx_leaf_nodes", int64[:],),
    ("output", types.Array(dtype=float64, ndim=2, layout="A"),),  ## config.floatX[:]
    ("leaf_node_value", float64),
])
class Tree:
    """Full binary tree.

    A full binary tree is a tree where each node has exactly zero or two children.
    This structure is used as the basic component of the Bayesian Additive Regression Tree (BART)

    Attributes
    ----------
    tree_structure : dict
        A dictionary that represents the nodes stored in breadth-first order, based in the array
        method for storing binary trees (https://en.wikipedia.org/wiki/Binary_tree#Arrays).
        The dictionary's keys are integers that represent the nodes position.
        The dictionary's values are objects of type Node that represent the split and leaf nodes
        of the tree itself.
    idx_leaf_nodes : list
        List with the index of the leaf nodes of the tree.
    output: array
        Array of shape number of observations, shape

    Parameters
    ----------
    leaf_node_value : int or float
    idx_data_points : array of integers
    num_observations : integer
    shape : int
    """

    def __init__(self, leaf_node_value, idx_data_points, output):
        self.tree_structure = {
            0: Node().new_leaf_node(0, leaf_node_value, idx_data_points)
        }
        self.idx_leaf_nodes = np.zeros(1, int64)
        self.output = output
        self.leaf_node_value = leaf_node_value

    def __getitem__(self, index):
        return self.get_node(index)

    def __setitem__(self, index, node):
        self.set_node(index, node)

    def __getstate__(self):
        return (self.tree_structure, self.idx_leaf_nodes, self.output, self.leaf_node_value)

    def __setstate__(self, state):
        self.tree_structure, self.idx_leaf_nodes, self.output, self.leaf_node_value = state

    def __reduce__(self):
        return (Tree, (self.tree_structure, self.idx_leaf_nodes, self.output, self.leaf_node_value))

    def copy(self):
        tree = Tree(self.leaf_node_value, self.tree_structure[0].idx_data_points, self.output.copy())
        for k, v in self.tree_structure.items():
            node = Node()
            node.index = v.index
            node.value = v.value
            node.idx_data_points = v.idx_data_points.copy()
            node.idx_split_variable = v.idx_split_variable
            tree[k] = node
        return tree

    def get_node(self, index) -> "Node":
        return self.tree_structure[index]

    def set_node(self, index, node):
        self.tree_structure[index] = node
        if node.is_leaf_node():
            np.append(self.idx_leaf_nodes, index)

    def delete_leaf_node(self, index):
        self.idx_leaf_nodes.remove(index)
        del self.tree_structure[index]

    def trim(self):
        a_tree = self.copy()
        del a_tree.output
        del a_tree.idx_leaf_nodes
        for k in a_tree.tree_structure.keys():
            current_node = a_tree[k]
            if current_node.is_leaf_node():
                del current_node.idx_data_points
        return a_tree

    def get_split_variables(self):
        return [
            node.idx_split_variable for node in self.tree_structure.values() if node.is_split_node()
        ]

    def _predict(self):
        output = self.output
        for node_index in self.idx_leaf_nodes:
            leaf_node = self.get_node(node_index)
            output[leaf_node.idx_data_points] = leaf_node.value
        return output.T

    def predict(self, x, excluded=None):
        """
        Predict output of tree for an (un)observed point x.

        Parameters
        ----------
        x : numpy array
            Unobserved point
        excluded: list
                Indexes of the variables to exclude when computing predictions

        Returns
        -------
        float
            Value of the leaf value where the unobserved point lies.
        """
        if excluded is None:
            excluded = []
        return self._traverse_tree(x, 0, excluded)

    def _traverse_tree(self, x, node_index, excluded):
        """
        Traverse the tree starting from a particular node given an unobserved point.

        Parameters
        ----------
        x : np.ndarray
        node_index : int

        Returns
        -------
        Leaf node value or mean of leaf node values
        """
        current_node = self.get_node(node_index)
        if current_node.is_leaf_node():
            return current_node.value
        if current_node.idx_split_variable in excluded:
            leaf_values = []
            self._traverse_leaf_values(leaf_values, node_index)
            return np.mean(leaf_values, 0)

        if x[current_node.idx_split_variable] <= current_node.value:
            left_child = current_node.get_idx_left_child()
            return self._traverse_tree(x, left_child, excluded)
        else:
            right_child = current_node.get_idx_right_child()
            return self._traverse_tree(x, right_child, excluded)

    def _traverse_leaf_values(self, leaf_values, node_index):
        """
        Traverse the tree appending leaf values starting from a particular node.

        Parameters
        ----------
        node_index : int

        Returns
        -------
        List of leaf node values
        """
        node = self.get_node(node_index)
        if node.is_leaf_node():
            np.append(leaf_values, node.value)
        else:
            self._traverse_leaf_values(leaf_values, node.get_idx_left_child())
            self._traverse_leaf_values(leaf_values, node.get_idx_right_child())
