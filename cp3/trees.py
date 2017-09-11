from math import log


def calc_shannon_ent(dataset):
    """
    get the shannon ent
    """
    entity_count = len(dataset)
    label_count = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in label_count.keys():
            label_count[current_label] = 0
        label_count[current_label] += 1

    print(label_count)

    shannon_ent = 0.0
    for key in label_count:
        prob = float(label_count[key]) / entity_count
        print(prob)
        shannon_ent -= prob * log(prob, 2)
        print(shannon_ent)

    return shannon_ent


def create_dataset():
    """
    create a sample dataset
    """
    dataset = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def split_dataset(dataset, axis, value):
    """
    split the dataset
    """
    results = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduce_feat_vec = feat_vec[:axis]
            reduce_feat_vec.extend(feat_vec[axis+1:])
            results.append(reduce_feat_vec)

    return results


def choose_bestfeature(dataset):
    """
    choose best feature to split data
    """
    feature_count = len(dataset[0]) - 1

    base_ent = calc_shannon_ent(dataset)
    best_info_gain = 0.0
    best_feature = -1

    for i in range(feature_count):
        feat_list = [example[i] for example in dataset]
        unique_vals = set(feat_list)
        new_ent = 0.0
        for value in unique_vals:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_ent += prob * calc_shannon_ent(sub_dataset)

        info_gain = base_ent - new_ent
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i

    return best_feature


def majority_cnt(class_list):
    """
    marjor cnt for no-category object
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0

        class_count[vote] += 1

    print(class_count)
    sorted_class_count = sorted(class_count.items(), key=lambda x: x[0], revere=True)
    print(sorted_class_count)
    return sorted_class_count[0][0]


def create_tree(dataset, labels):
    """
    create tree
    """
    class_list = [data[-1] for data in dataset]
    # category finish
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # no feature to split, use most count
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)

    best_feat = choose_bestfeature(dataset)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del(labels[best_feat])

    feat_values = [example[best_feat] for example in dataset]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sublabels = labels[:]
        my_tree[best_feat_label][value] = create_tree(
            split_dataset(dataset, best_feat, value), sublabels)

    print(my_tree)
    return my_tree
