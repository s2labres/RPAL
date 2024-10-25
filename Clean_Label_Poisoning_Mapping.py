import pickle

from RPAL import data, loader, poison

'''
Clean Label Attack Sample Generation
'''

# Setup control variables
random_state = 999

# Percent of points to retrain on each month
al_strengths = [str(2 ** x) for x in range(0, 5)]
al_strengths[0] = str(0)

# Percents of generation malware set to be added as poisoning points
poisoning_strength = 0.16
p_settings = [0.02, 0.04, 0.08, 0.16]
p_amounts = [1156, 2312, 4626, 9252]

# File path for dataset
file_path = 'Dataset/extended-features'
extension = '.p'
X_extension = 'reduced'

# Underlying malware distribution
mw_distribution = 0.1

def_dist = 100000


def main():
    print("Starting....")

    # Load dataset
    print("Loading Data...")
    X, y, t = loader.load_p(fname=file_path, X_extension=X_extension)

    if X_extension == 'full':
        X = data.format_features(X)

    # Partition dataset
    print("Splitting Dataset....")
    splits = data.prepare_splits(X, y, t, random_state)

    results = poison.label_flip_feature_mapping(p_settings=p_settings,
                                                p_amounts=p_amounts,
                                                splits=splits,
                                                mw_distribution=mw_distribution,
                                                file_path=file_path,
                                                random_state=random_state)

    with open(file_path + '-Label-Flip-Poison-Feature-Mapping.p', 'wb') as fp:
        pickle.dump(results, fp)


if __name__ == '__main__':
    main()
