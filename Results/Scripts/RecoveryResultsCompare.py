import pickle


def main():
    deviations = [0.05, 0.01, 0.00]

    ft1 = "Drebin-Label-Flip-SVM-Tesseract"
    ft2 = "Drebin-Label-Flip-Deep-Tesseract"
    ft3 = "Drebin-Label-Flip-RF-Tesseract"

    # f1 better, f2 better, f3 better, mixed
    results = [[[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]]
    results2 = [0, 0, 0, 0]

    for d in deviations:

        temp_results = [[[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]]

        recovery1 = pickle.load(
            open("../Data/RecoveryData/" + ft1 + "-Tolerance-" + str(int(d * 100)) + ".p", "rb"))
        recovery2 = pickle.load(
            open("../Data/RecoveryData/" + ft2 + "-Tolerance-" + str(int(d * 100)) + ".p", "rb"))
        recovery3 = pickle.load(
            open("../Data/RecoveryData/" + ft3 + "-Tolerance-" + str(int(d * 100)) + ".p", "rb"))

        recovery = [recovery1, recovery2, recovery3]

        for x in range(len(recovery)):
            for y in range(len(recovery)):
                if x != y:
                    for i in range(len(recovery1)):
                        for j in range(len(recovery1[i])):
                            r1 = recovery[x][i][j]
                            r2 = recovery[y][i][j]

                            if r1[0] == -1:
                                r1 = (999, 0)

                            if r2[0] == -1:
                                r2 = (999, 0)

                            if ((r1[0] == r2[0] and r1[1] == r2[1])
                                    or (r1[0] < r2[0] and r1[1] < r2[1])
                                    or (r1[0] > r2[0] and r1[1] > r2[1])):
                                results[x][y][1] += 1
                                temp_results[x][y][1] += 1
                            elif ((r1[0] < r2[0] and r1[1] > r2[1])
                                  or (r1[0] < r2[0] and r1[1] == r2[1])
                                  or (r1[0] == r2[0] and r1[1] > r2[1])):
                                results[x][y][0] += 1
                                temp_results[x][y][0] += 1

        for i in range(len(recovery1)):
            for j in range(len(recovery1[i])):
                r1 = recovery1[i][j]
                r2 = recovery2[i][j]
                r3 = recovery3[i][j]

                if r1[0] == -1:
                    r1 = (999, 0)

                if r2[0] == -1:
                    r2 = (999, 0)

                if r3[0] == -1:
                    r3 = (999, 0)

                mixed = False
                r1better = False
                r2better = False

                if ((r1[0] == r2[0] and r1[1] == r2[1])
                        or (r1[0] < r2[0] and r1[1] < r2[1])
                        or (r1[0] > r2[0] and r1[1] > r2[1])):
                    mixed = True
                elif ((r1[0] < r2[0] and r1[1] > r2[1])
                      or (r1[0] < r2[0] and r1[1] == r2[1])
                      or (r1[0] == r2[0] and r1[1] > r2[1])):
                    r1better = True
                elif ((r2[0] < r1[0] and r2[1] > r1[1])
                      or (r2[0] < r1[0] and r2[1] == r1[1])
                      or (r2[0] == r1[0] and r2[1] > r1[1])):
                    r2better = True

                if mixed:
                    results2[3] += 1
                elif r1better:
                    if ((r1[0] == r3[0] and r1[1] == r3[1])
                            or (r1[0] < r3[0] and r1[1] < r3[1])
                            or (r1[0] > r3[0] and r1[1] > r3[1])):
                        results2[3] += 1
                    elif ((r1[0] < r3[0] and r1[1] > r3[1])
                          or (r1[0] < r3[0] and r1[1] == r3[1])
                          or (r1[0] == r3[0] and r1[1] > r3[1])):
                        results2[0] += 1
                    elif ((r3[0] < r1[0] and r3[1] > r1[1])
                          or (r3[0] < r1[0] and r3[1] == r1[1])
                          or (r3[0] == r1[0] and r3[1] > r1[1])):
                        results2[2] += 1
                elif r2better:
                    if ((r2[0] == r3[0] and r2[1] == r3[1])
                            or (r2[0] < r3[0] and r2[1] < r3[1])
                            or (r2[0] > r3[0] and r2[1] > r3[1])):
                        results2[3] += 1
                    elif ((r2[0] < r3[0] and r2[1] > r3[1])
                          or (r2[0] < r3[0] and r2[1] == r3[1])
                          or (r2[0] == r3[0] and r2[1] > r3[1])):
                        results2[1] += 1
                    elif ((r3[0] < r2[0] and r3[1] > r2[1])
                          or (r3[0] < r2[0] and r3[1] == r2[1])
                          or (r3[0] == r2[0] and r3[1] > r2[1])):
                        results2[2] += 1

        print('Deviation ' + str(d) + ' results:')
        print(temp_results)

        for r in recovery:
            print("average recovery rate: ")
            sum_r = 0
            count_r = 0
            for i in r:
                for j in i:
                    if j[1] >= 0:
                        sum_r += j[1]
                        count_r += 1

            print(str(sum_r / count_r))

    print('Complete Results: ')
    print(results)
    print(results2)
    print()


if __name__ == '__main__':
    main()
