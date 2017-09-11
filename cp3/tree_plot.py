# pylint: disable=invalid-name

import matplotlib.pyplot as plt


decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plot_node(node_txt, center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(node_txt,
                             xy=parent_pt,
                             xycoords="axes fraction",
                             xytext=center_pt,
                             textcoords="axes fractin",
                             va="center",
                             ha="center",
                             bbox=node_type,
                             arrowprops=arrow_args)


def create_plot():
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)
    plot_node("决策节点", (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node("叶节点", (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


def get_num_leafs(my_tree):
    """
    get the leafs count
    """
    leaf_num = 0
    first_str = my_tree.keys()[0]
    second_dict = my_tree[first_str]

    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            leaf_num += get_num_leafs(second_dict[key])
        else:
            leaf_num += 1

    return leaf_num


def get_tree_depth(my_tree):
    max_depth = 0
    first_str = my_tree.keys()[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth

    return this_depth


def test_tree(i):
    """
    get a tree for test
    """
    trees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]

    return trees[i]
