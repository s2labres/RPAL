import json
import pickle

from RPAL import grapher

from collections import defaultdict

"""
Generate Recovery Plots
"""


def main():
    deviation = 0.05

    P = "8"

    ft = "Drebin-Label-Flip-RF-Tesseract"

    fname = "../Data/" + ft + "/" + ft

    labels = ["P - " + P + "\%",
              "P - " + P + "\% / AL - 2\%",
              "P - " + P + "\% / AL - 4\%",
              "P - " + P + "\% / AL - 8\%",
              "P - " + P + "\% / AL - 16\%"]

    colors = ["black", "orange", "blue", "red", "green"]

    markers = ["P", "d", "s", "X", "o"]

    save_fname = "../Data/RecoveryData/" + ft + "-P" + P + "-T" + str(int(deviation*100)) + ".pdf"

    files = [fname + "-AL0-P" + P + ".json",
             fname + "-AL2-P" + P + ".json",
             fname + "-AL4-P" + P + ".json",
             fname + "-AL8-P" + P + ".json",
             fname + "-AL16-P" + P + ".json"]

    data = [None for x in files]

    for i in range(0, len(files)):
        data_file = open(files[i])
        data[i] = defaultdict(list, json.load(data_file))
        data_file.close()

    grapher.recovery_plot(data, labels, colors, markers, False, save_fname)


if __name__ == '__main__':
    main()
