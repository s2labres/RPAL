import json
import pickle

from RPAL import grapher

from collections import defaultdict

"""
Generate Recovery Plots
"""


def main():
    ft1 = "SVM-Tesseract"
    ft2 = "Deep-Tesseract"
    ft3 = "RF-Tesseract"

    fname1 = "../Data/Drebin-Label-Flip-" + ft1 + "/Drebin-Label-Flip-" + ft1 + "-AL0-P0.json"
    fname2 = "../Data/Drebin-Label-Flip-" + ft2 + "/Drebin-Label-Flip-" + ft2 + "-AL0-P0.json"
    fname3 = "../Data/Drebin-Label-Flip-" + ft3 + "/Drebin-Label-Flip-" + ft3 + "-AL0-P0.json"

    labels = ["SVM - D1418",
              "DNN - D1418",
              "RF - D1418"]

    colors = ["green", "blue", "red"]

    markers = ["o", "s", "^"]

    save_fname = "../Data/RecoveryData/Baselines.pdf"

    files = [fname1, fname2, fname3]

    data = [None for x in files]

    for i in range(0, len(files)):
        data_file = open(files[i])
        data[i] = defaultdict(list, json.load(data_file))
        data_file.close()

    grapher.recovery_plot(data, labels, colors, markers, False, save_fname)


if __name__ == '__main__':
    main()
