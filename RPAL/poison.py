import pickle
from copy import deepcopy
import random
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import multiprocessing as mp


def label_flip_poison_by_modification(splits, poisoning_strength, mw_distribution, random_state):
    """ Returns a poisoned version of splits

        Args:
            splits (np.ndarray, list, np.ndarray, list, np.ndarray, list):
                Training partition of predictors X.
                List of testing partitions of predictors X.

                Training partition of output variables y.
                List of testing partitions of predictors y.

                Training partition of timestamps variables t.
                List of testing partitions of timestamps t.

            poisoning_strength (float): percent of malware to add a flipped point

            mw_distribution (float): percent of malware in dataset

            random_state (int): seed for random number generator

        Returns:
            poisoned_splits (np.ndarray, list, np.ndarray, list, np.ndarray, list):
                Poisoned Training partition of predictors X.
                List of testing partitions of predictors X.

                Poisoned Training partition of output variables y.
                List of testing partitions of predictors y.

                Poisoned Training partition of timestamps variables t.
                List of testing partitions of timestamps t.
    """

    poisoned_splits = deepcopy(splits)

    if poisoning_strength == float(0):
        return poisoned_splits, []

    random.seed(random_state)

    poison_indexes = []

    # Per class indexes
    gw_indexes = (np.where(poisoned_splits[2] == 0))[0]
    mw_indexes = (np.where(poisoned_splits[2] == 1))[0]

    # Randomize Indexes
    random.shuffle(gw_indexes)
    random.shuffle(mw_indexes)

    # Percent of poisoning per class
    if poisoning_strength > (2*mw_distribution):
        gw_percent = (poisoning_strength - mw_distribution) / (1 - mw_distribution)
    else:
        gw_percent = (poisoning_strength / 2) / (1 - mw_distribution)

    gw_count = int(len(gw_indexes)*gw_percent)
    mw_count = gw_count if gw_count <= len(mw_indexes) else len(mw_indexes)

    # Goodware label flip
    for i in range(0, gw_count):
        index = gw_indexes[i]
        y = poisoned_splits[2][index]
        poisoned_splits[2][index] = int(not bool(y))
        poison_indexes.append(index)

    # Malware label flip
    for j in range(0, mw_count):
        index = mw_indexes[j]
        y = poisoned_splits[2][index]
        poisoned_splits[2][index] = int(not bool(y))
        poison_indexes.append(index)

    return poisoned_splits, poison_indexes


def label_flip_poison_by_addition(training_splits, generation_splits, poisoning_strength, random_state):
    """ Returns a poisoned version of splits

        Args:
            training_splits (np.ndarray, list, np.ndarray, list, np.ndarray, list):
                Training partition of predictors X.
                List of testing partitions of predictors X.

                Training partition of output variables y.
                List of testing partitions of predictors y.

                Training partition of timestamps variables t.
                List of testing partitions of timestamps t.

            generation_splits (np.ndarray, list, np.ndarray, list, np.ndarray, list):
                Training partition of predictors X.
                List of testing partitions of predictors X.

                Training partition of output variables y.
                List of testing partitions of predictors y.

                Training partition of timestamps variables t.
                List of testing partitions of timestamps t.

            poisoning_strength (float): percent of malware to add a flipped point

            random_state (int): seed for random number generator

        Returns:
            poisoned_splits (np.ndarray, list, np.ndarray, list, np.ndarray, list):
                Poisoned Training partition of predictors X.
                List of testing partitions of predictors X.

                Poisoned Training partition of output variables y.
                List of testing partitions of predictors y.

                Poisoned Training partition of timestamps variables t.
                List of testing partitions of timestamps t.
    """

    splits = deepcopy(training_splits)

    if poisoning_strength == float(0):
        return splits

    random.seed(random_state)
    num_labels = len(generation_splits[2])

    indexes = [y for y in range(0, num_labels)]
    random.shuffle(indexes)

    X_pois = generation_splits[0][indexes[0]]
    y_pois = [int(not bool(generation_splits[2][indexes[0]]))]
    t_pois = [generation_splits[4][indexes[0]]]

    for i in range(1, int(num_labels*poisoning_strength)):
        index = indexes[i]

        X_pois = sp.vstack((X_pois, generation_splits[0][index]))
        y_pois.append(int(not bool(generation_splits[2][index])))
        t_pois.append(generation_splits[4][index])

    X_train, X_tests, y_train, y_tests, t_train, t_tests = splits

    return (sp.vstack((X_train, X_pois)),
            X_tests,
            np.hstack((y_train, y_pois)),
            y_tests,
            np.hstack((t_train, t_pois)),
            t_tests)


def distance_from_base(def_dist, X_train, y_train, b):
    distances = [def_dist for x in range(len(y_train))]
    target_distances = [def_dist for x in range(len(y_train))]
    base_distances = [def_dist for x in range(len(y_train))]

    for t in tqdm(range(len(y_train)), desc='Target'):
        if b != t and y_train[t] != y_train[b]:
            target_distances[t] = abs(sum([x for x in (X_train[b] - X_train[t]).toarray()[0] if x < 0]))
            base_distances[t] = abs(sum([x for x in (X_train[t] - X_train[b]).toarray()[0] if x < 0]))
            distance = max(target_distances[t] + base_distances[t], 0)
            distances[t] = distance

    return b, distances, target_distances, base_distances


def label_flip_feature_mapping(p_settings, p_amounts, splits, mw_distribution, file_path, random_state):
    splits = deepcopy(splits)

    random.seed(random_state)

    pois_indexes = []

    # Per class indexes
    gw_indexes = (np.where(splits[2] == 0))[0]
    mw_indexes = (np.where(splits[2] == 1))[0]

    # Randomize Indexes
    random.shuffle(gw_indexes)
    random.shuffle(mw_indexes)

    # Percent of poisoning per class
    max_poisoning_strength = p_settings[-1]
    if max_poisoning_strength > (2 * mw_distribution):
        gw_percent = (max_poisoning_strength - mw_distribution) / (1 - mw_distribution)
        mw_percent = 1
    else:
        gw_percent = (max_poisoning_strength / 2) / (1 - mw_distribution)
        mw_percent = (max_poisoning_strength / 2) / mw_distribution

    X_train = splits[0][gw_indexes[0]]
    y_train = [splits[2][gw_indexes[0]]]
    pois_indexes = [gw_indexes[0]]

    gw_count = int(len(gw_indexes) * gw_percent)
    mw_count = int(len(mw_indexes) * mw_percent)
    extra_count = gw_count - mw_count

    # Goodware Sample Collection
    for i in range(1, gw_count):
        index = gw_indexes[i]
        X_train = sp.vstack((X_train, splits[0][index]))
        y_train.append(splits[2][index])
        pois_indexes.append(index)

    # Malware Sample Collection
    for i in range(0, gw_count):
        index = mw_indexes[i]
        X_train = sp.vstack((X_train, splits[0][index]))
        y_train.append(splits[2][index])
        pois_indexes.append(index)

    n_samples = len(y_train)

    print('Getting Distance...')
    try:
        distances = pickle.load(open(file_path + '-Label-Flip-Poison-Distances-' + str(random_state) + '.p', 'rb'))

    except FileNotFoundError:
        # Distance Calculation
        pool = mp.Pool(mp.cpu_count())
        distances = pool.starmap(distance_from_base, [(X_train, y_train, b) for b in range(n_samples)])
        pool.close()
        with open(file_path + '-Label-Flip-Poison-Distances-' + str(random_state) + '.p', 'wb') as fp:
            pickle.dump(distances, fp)

    target_distances = [x[2] for x in distances]
    distances = [x[1] for x in distances]

    print('Selecting Targets...')
    try:
        targets = [pickle.load(open(file_path + '-Label-Flip-Poison-Targets-' + str(int(x*100)) + '-' + str(random_state)
                                    + '.p', 'rb')) for x in p_settings]

    except FileNotFoundError:
        targets = [[(-1, -1) for y in range(p_amounts[x])] for x in range(len(p_settings))]

        for p in range(len(p_settings)):
            print('Starting Setting: ' + str(p_settings[p]))

            try:
                targets[p] = pickle.load(open(file_path + '-Label-Flip-Poison-Targets-' +
                                              str(int(p_settings[p] * 100)) + '-' + str(random_state) + '.p', 'rb'))
            except FileNotFoundError:
                p_samples = p_amounts[p]
                p_distances = [[y for y in (x[:p_samples//2] + x[gw_count:(gw_count+p_samples//2)])]
                               for x in (distances[:p_samples//2] + distances[gw_count:(gw_count+p_samples//2)])]

                target_order = [np.argsort(p_distances[x]) for x in range(p_samples)]
                target_idx = [0 for x in range(p_samples)]

                assigned = {target_order[i][target_idx[i]] for i in range(p_samples//2)}

                conflict = True
                while conflict:
                    conflict = False
                    conflict_counter = 0
                    for i in tqdm(range(p_samples//2), desc='Conflict Search'):
                        conflicts = [x for x in range(p_samples//2)
                                     if target_order[i][target_idx[i]] == target_order[x][target_idx[x]]
                                     and i != x]

                        for c in conflicts:
                            conflict_counter += 1
                            for j in range(len(target_order[c])):
                                if target_order[c][j] not in assigned:
                                    target_idx[c] = j
                                    assigned.add(target_order[c][j])
                                    break

                        if conflicts:
                            conflict = True

                    print('Conflicts: ' + str(conflict_counter))
                    print('Avg Distance: ' + str(sum([p_distances[x][target_order[x][target_idx[x]]]*2 for x in range(p_samples//2)])/len(targets[p])))

                target_temp = [(target_order[x][target_idx[x]], p_distances[x][target_order[x][target_idx[x]]])
                               for x in range(p_samples//2)]

                for i in range(p_samples//2):
                    idx = target_temp[i][0]
                    val = target_temp[i][1]
                    targets[p][i] = target_temp[i]
                    targets[p][idx] = (i, val)

                with (open(file_path + '-Label-Flip-Poison-Targets-' + str(int(p_settings[p]*100)) + '-' +
                           str(random_state) + '.p', 'wb') as fp):
                    pickle.dump(targets[p], fp)

            print('Complete Setting: ' + str(p_settings[p]))

    return targets, pois_indexes, gw_count, mw_count
