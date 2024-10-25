import json
import pickle

from RPAL import recovery, grapher

from collections import defaultdict

"""
Extract Recovery Data and save it
"""


def main():
    deviation = 0.0

    ft = "Deep-Tesseract"

    fname = "../Data/Drebin-Label-Flip-" + ft + "/Drebin-Label-Flip-" + ft

    results = [[None for y in range(0, 5)] for x in range(1, 5)]

    for pois_rate in range(1, 5):
        for al_rate in range(0, 5):

            # Load Experiment Data
            if al_rate > 0:
                data_file = open(fname + "-AL" + str(2 ** al_rate) + "-P" + str(2 ** pois_rate) + ".json")
            else:
                data_file = open(fname + "-AL" + str(0) + "-P" + str(2 ** pois_rate) + ".json")
            exp_data = defaultdict(list, json.load(data_file))
            data_file.close()

            # Load Baseline Data
            if al_rate > 0:
                data_file = open(fname + "-AL" + str(2 ** al_rate) + "-P" + str(0) + ".json")
            else:
                data_file = open(fname + "-AL" + str(0) + "-P" + str(0) + ".json")
            base_data = defaultdict(list, json.load(data_file))
            data_file.close()

            # Extract recovery point
            p_index = (pois_rate - 1)
            al_index = al_rate
            results[p_index][al_index] = recovery.extract_recovery_data(experiment=exp_data,
                                                                        baseline=base_data,
                                                                        deviation=deviation)

    with open("../Data/RecoveryData/Drebin-Label-Flip-" + ft + "-Tolerance-" + str(int(deviation*100)) + ".p", "wb") as file:
        pickle.dump(results, file)


if __name__ == '__main__':
    main()
