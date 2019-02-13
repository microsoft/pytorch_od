# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import print_function


def read_softmax_tree(tree_file):
    """Simple parsing of softmax tree with subgroups
    :param tree_file: path to the tree file, or open file object
    :type tree_file: str or file
    """
    group_offsets = []
    group_sizes = []
    cid_groups = []
    parents = []
    child = []  # child group
    child_sizes = []  # number of child groups
    root_size = 0  # number of child sub-groups at root
    last_p = -1
    last_sg = -1
    groups = 0
    sub_groups = 0
    size = 0
    n = 0
    with open(tree_file, 'r') as f:
        for line in f.readlines():
            tokens = [t for t in line.split(' ') if t]
            assert len(tokens) == 2 or len(tokens) == 3, "invalid tree: {} node: {} line: {}".format(
                tree_file, n, line)
            p = int(tokens[1])
            assert n > p >= -1, "invalid parent: {} node: {} tree: {}".format(p, n, tree_file)
            parents.append(p)
            sg = -1
            if len(tokens) == 3:
                sg = int(tokens[2])
            new_group = new_sub_group = False
            if p != last_p:
                last_p = p
                last_sg = sg
                new_group = True
                sub_groups = 0
            elif sg != last_sg:
                assert sg > last_sg, "invalid sg: {} node: {} tree: {}".format(sg, n, tree_file)
                last_sg = sg
                new_sub_group = True
                sub_groups += 1
            if new_group or new_sub_group:
                group_sizes.append(size)
                group_offsets.append(n - size)
                groups += 1
                size = 0
            child.append(-1)
            child_sizes.append(0)
            if p >= 0:
                if new_group:
                    assert child[p] == -1, "node: {} parent discontinuity in tree: {}".format(n, tree_file)
                    child[p] = groups  # start group of child subgroup
                elif new_sub_group:
                    child_sizes[p] = sub_groups
            else:
                root_size = sub_groups
            n += 1
            size += 1
            cid_groups.append(groups)
    group_sizes.append(size)
    group_offsets.append(n - size)

    assert len(cid_groups) == len(parents) == len(child) == len(child_sizes)
    assert len(group_offsets) == len(group_sizes) == max(cid_groups) + 1
    return group_offsets, group_sizes, cid_groups, parents, child, child_sizes, root_size
