import json
import pickle
from collections import defaultdict


def main():
    fixed_pois = 8
    fixed_al = 8

    ft1 = "Drebin-Label-Flip-SVM-Tesseract"
    ft2 = "Drebin-Label-Flip-Deep-Tesseract"
    ft3 = "Drebin-Label-Flip-RF-Tesseract"

    fname1 = "../Data/" + ft1 + "/" + ft1
    fname2 = "../Data/" + ft2 + "/" + ft2
    fname3 = "../Data/" + ft3 + "/" + ft3

    # exp1 better, exp2 better, exp3 better, mixed
    results = [0, 0, 0, 0]

    for pois_rate in range(1, 5):
        # Load Experiment1 Data
        data_file = open(fname1 + "-AL" + str(fixed_al) + "-P" + str(2 ** pois_rate) + ".json")
        exp1_data = defaultdict(list, json.load(data_file))
        data_file.close()
        exp1_f1 = exp1_data['f1']

        # Load Experiment2 Data
        data_file = open(fname2 + "-AL" + str(fixed_al) + "-P" + str(2 ** pois_rate) + ".json")
        exp2_data = defaultdict(list, json.load(data_file))
        data_file.close()
        exp2_f1 = exp2_data['f1']

        # Load Experiment3 Data
        data_file = open(fname3 + "-AL" + str(fixed_al) + "-P" + str(2 ** pois_rate) + ".json")
        exp3_data = defaultdict(list, json.load(data_file))
        data_file.close()
        exp3_f1 = exp3_data['f1']

        for i in range(len(exp1_f1)):
            if exp1_f1[i] > exp2_f1[i] and exp1_f1[i] > exp3_f1[i]:
                results[0] += 1
            elif exp2_f1[i] > exp1_f1[i] and exp2_f1[i] > exp3_f1[i]:
                results[1] += 1
            elif exp3_f1[i] > exp1_f1[i] and exp3_f1[i] > exp2_f1[i]:
                results[2] += 1
            else:
                results[3] += 1

    print("Fixed Poisoning Comparison: ")
    print(results)

    # exp1 better, exp2 better, mixed
    results = [0, 0, 0, 0]

    for al_rate in range(1, 5):
        # Load Experiment1 Data
        data_file = open(fname1 + "-AL" + str(2 ** al_rate) + "-P" + str(fixed_pois) + ".json")
        exp1_data = defaultdict(list, json.load(data_file))
        data_file.close()
        exp1_f1 = exp1_data['f1']

        # Load Experiment2 Data
        data_file = open(fname2 + "-AL" + str(2 ** al_rate) + "-P" + str(fixed_pois) + ".json")
        exp2_data = defaultdict(list, json.load(data_file))
        data_file.close()
        exp2_f1 = exp2_data['f1']

        # Load Experiment3 Data
        data_file = open(fname3 + "-AL" + str(2 ** al_rate) + "-P" + str(fixed_pois) + ".json")
        exp3_data = defaultdict(list, json.load(data_file))
        data_file.close()
        exp3_f1 = exp3_data['f1']

        for i in range(len(exp1_f1)):
            if exp1_f1[i] > exp2_f1[i] and exp1_f1[i] > exp3_f1[i]:
                results[0] += 1
            elif exp2_f1[i] > exp1_f1[i] and exp2_f1[i] > exp3_f1[i]:
                results[1] += 1
            elif exp3_f1[i] > exp1_f1[i] and exp3_f1[i] > exp2_f1[i]:
                results[2] += 1
            else:
                results[2] += 1

    print("Fixed Active Learning Comparison: ")
    print(results)


if __name__ == '__main__':
    main()