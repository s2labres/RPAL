import json
import os
import pickle

from RPAL import grapher

from collections import defaultdict

"""
Generate Recovery Plots
"""


def main():
    os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/universal-darwin'

    deviation = 0.05

    AL = "8"

    ft = "Drebin-Label-Flip-Deep-Tesseract"

    fname = "../Data/" + ft + "/" + ft

    labels = ["AL - " + AL + "\%",
              "Tolerance Margin",
              "AL - " + AL + "\% / P - 2\%",
              "AL - " + AL + "\% / P - 4\%",
              "AL - " + AL + "\% / P - 8\%",
              "AL - " + AL + "\% / P - 16\%"]

    colors = ["black", "gray", "orange", "blue", "red", "green"]

    markers = ["P", "", "d", "s", "X", "o"]

    save_fname = "../Data/RecoveryData/" + ft + "-AL" + AL + "-T" + str(int(deviation * 100)) + ".pdf"

    files = [fname + "-AL" + AL + "-P0.json",
             fname + "-AL" + AL + "-P0.json",
             fname + "-AL" + AL + "-P2.json",
             fname + "-AL" + AL + "-P4.json",
             fname + "-AL" + AL + "-P8.json",
             fname + "-AL" + AL + "-P16.json"]

    data = [None for x in files]

    for i in range(0, len(files)):
        data_file = open(files[i])
        data[i] = defaultdict(list, json.load(data_file))
        data_file.close()

    for f in range(len(data[1]['f1'])):
        score = data[1]['f1'][f]
        if score > deviation:
            data[1]['f1'][f] = score - deviation
        else:
            data[1]['f1'][f] = 0

    grapher.recovery_plot(data, labels, colors, markers, True, save_fname)


if __name__ == '__main__':
    main()
