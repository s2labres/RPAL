import json
from numpyencoder import NumpyEncoder

from RPAL import data, classification, poison, loader
from deepdrebin import DeepDrebin

'''
Drebin using DNN 2014-2018 Experiment
Utilizing Label Flip Attack
'''

# If a batch or a single run
is_batch = True

# For Single Runs
p_single = 0.0  # Percent as decimal
al_single = '0'  # Percent as str

# For Multiple Runs
p_skip = -1
al_skip = -1

# Setup control variables
random_state = 999

# Percent of points to retrain on each month
al_strengths = [str(2 ** x) for x in range(0, 5)]
al_strengths[0] = str(0)

# Percents of generation malware set to be added as poisoning points
poisoning_strengths = [(2 ** x) / 100 for x in range(0, 5)]
poisoning_strengths[0] = 0.0

# File path for dataset
file_path = 'Dataset/extended-features'
extension = '.p'
X_extension = 'reduced'

# Underlying malware distribution
mw_distribution = 0.1


def run_exp(poisoning_strength, al_strength, splits, random_state, X, y, t):
    print("Running Attack With "
          + str(int(poisoning_strength * 100)) + "% Poisoning & "
          + str(al_strength) + "% Active Learning...")

    # Generation of poisoned splits
    poisoned_splits, _ = poison.label_flip_poison_by_modification(splits=splits,
                                                                  poisoning_strength=poisoning_strength,
                                                                  mw_distribution=mw_distribution,
                                                                  random_state=random_state)

    # Perform evaluation
    print("Evaluating....")
    desc = 'AL' + al_strength + '-P' + str(int(poisoning_strength * 100))
    clf = DeepDrebin()
    clf.setup(desc=desc, train_time=2014, X=X, y=y, t=t)
    results = classification.classify(clf=clf, splits=poisoned_splits, sampling=al_strength)

    # Save results in json
    with open("Results/Data/Drebin-Label-Flip-Deep-Tesseract/Drebin-Label-Flip-Deep-Tesseract-AL" + al_strength
              + "-P" + str(int(poisoning_strength * 100)) + ".json", "w") as output_file:
        json_results = json.dumps(results, cls=NumpyEncoder)
        output_file.write(json_results)

    return 0


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

    if is_batch:
        # Running attack combinations
        print("Running attack combinations...")
        for poisoning_strength in poisoning_strengths:
            for al_strength in al_strengths:
                if (poisoning_strength > p_skip) or (poisoning_strength == p_skip and int(al_strength) > al_skip):
                    run_exp(poisoning_strength=poisoning_strength,
                            al_strength=al_strength,
                            splits=splits,
                            random_state=random_state,
                            X=X, y=y, t=t)

    else:
        run_exp(poisoning_strength=p_single,
                al_strength=al_single,
                splits=splits,
                random_state=random_state)


if __name__ == '__main__':
    main()
