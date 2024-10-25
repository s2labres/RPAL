import matplotlib.pyplot as plt
import pandas as pd

from tesseract import metrics, viz


def plot(results, save_fname):
    """ plots the results on a graph and saves the graph to location/name provided
        Modified version of Tesseract viz.plot_decay function
        Args:
            results: results returned from classification
                see classification.classify()

            save_fname: location and name of the result graph to be saved

        Returns:
            float: the AUT score
    """
    print("Plotting...")

    # View results
    metrics.print_metrics(results)

    # View AUT(F1, 24 months) as a measure of robustness over time
    aut_score = metrics.aut(results, 'f1')
    aut_score_str = "AUT: " + str(aut_score)
    print(aut_score_str)

    output = viz.plot_decay(results, titles=(aut_score_str, ""))
    output.savefig(save_fname)

    return aut_score


def multi_plot(results, labels, colors, markers, title, save_fname):
    """ Plots the multiple results on a graph and saves the graph to save_fname.
            Args:
                results (list): list of results returned from classification
                    see classification.classify()

                labels (list): List of the legend label for each data set

                colors (list): list of the colour for f1 score of each dataset

                markers (list): List of markers to use f1 score of each dataset

                title (str): title of the figure

                save_fname (str): location and name of the result graph to be saved

            Returns:
                list: list of the AUT score
        """
    print("Plotting...")

    aut_scores = []

    # View results
    for r in results:
        metrics.print_metrics(r)

        # View AUT(F1, 24 months) as a measure of robustness over time
        aut_score = metrics.aut(r, 'f1')
        aut_scores = aut_scores + [aut_score]

        aut_score_str = "AUT: " + str(aut_score)
        print(aut_score_str)

    # Convert Into Dataframe (Bug Fix From Tesseract viz.plot_decay)
    for i in range(len(results)):
        del results[i]['auc_roc']  # Otherwise hampers the DataFrame conversion
        results[i] = pd.DataFrame(dict(results[i]),
                                  index=range(1, len(results[i]['f1']) + 1))

    viz.set_style()

    fig, axes = plt.subplots(1, 1)
    axes = axes if hasattr(axes, '__iter__') else (axes,)

    axes[0].set_title(title)

    for i in range(0, len(results)):
        viz.plot_f1(axes[0], results[i], 0.4, label=labels[i], color=colors[i], marker=markers[i])

    # If Baseline fill under
    series = results[0]['f1']
    axes[0].fill_between(results[0].index,
                         series,
                         alpha=0.2,
                         facecolor='lightgrey',
                         hatch='//',
                         edgecolor='red')

    # Legend
    viz.add_legend(axes[0], loc="best")

    viz.style_axes(axes, len(results[0]['f1']))
    axes[0].set_ylabel("F1 Score")
    fig.set_size_inches(7, 5)
    plt.tight_layout()

    plt.savefig(save_fname)

    return aut_scores


def recovery_plot(results, labels, colors, markers, isMargin, save_fname):
    """ Plots the multiple results on a graph and saves the graph to save_fname.
            Args:
                results (list): results for baseline, tolerance margin, variable data

                labels (list): List of the legend label for each data set

                colors (list): list of the colour for f1 score of each dataset

                markers (list): List of markers to use f1 score of each dataset

                isMargin (bool): if the second result is margin

                save_fname (str): location and name of the result graph to be saved

            Returns:
                list: list of the AUT score
        """
    print("Plotting...")

    aut_scores = []

    # View results
    for r in results:
        metrics.print_metrics(r)

        # View AUT(F1, 24 months) as a measure of robustness over time
        aut_score = metrics.aut(r, 'f1')
        aut_scores = aut_scores + [aut_score]

        aut_score_str = "AUT: " + str(aut_score)
        print(aut_score_str)

    # Convert Into Dataframe (Bug Fix From Tesseract viz.plot_decay)
    for i in range(len(results)):
        del results[i]['auc_roc']  # Otherwise hampers the DataFrame conversion
        results[i] = pd.DataFrame(dict(results[i]),
                                  index=range(1, len(results[i]['f1']) + 1))

    viz.set_style()

    fig, axes = plt.subplots(1, 1)
    axes = axes if hasattr(axes, '__iter__') else (axes,)

    axes[0].set_title("")

    for i in range(len(results)):
        if i == 0 and False:
            axes[0].plot(results[i].index, results[i]['f1'],
                         label=labels[i],
                         alpha=1,
                         marker=markers[i],
                         c=colors[i],
                         markeredgewidth=1,
                         linewidth=1,
                         markersize=4,
                         linestyle='dashed')
        else:
            axes[0].plot(results[i].index, results[i]['f1'],
                         label=labels[i],
                         alpha=1,
                         marker=markers[i],
                         c=colors[i],
                         markeredgewidth=1,
                         linewidth=1,
                         markersize=4)

    if isMargin:
        axes[0].fill_between(results[0].index,
                             y1=results[0]['f1'],
                             y2=results[1]['f1'],
                             alpha=0.5,
                             facecolor='lightgrey',
                             hatch='//',
                             edgecolor='darkgrey')

    # Legend
    viz.add_legend(axes[0], loc="lower right")

    viz.style_axes(axes, len(results[0]['f1']))
    axes[0].set_ylabel("F1 Score")
    fig.set_size_inches(5, 4)
    plt.tight_layout()

    plt.savefig(save_fname, transparent=True)

    return aut_scores


def plot_batch(sampling, poisoning, scores, sampling_title='Sampling',
               poisoning_title='Poisoning', y_title="AUT", title='Batch Results'):
    """ Plots batch result data to an interactive 3d plot
        Args:
            sampling (list): sorted list of poisoning point values
                (note include 0)

            poisoning (list): sorted list of active learning sampling values
                (note include 0)

            scores (list): AUT Score values sorted by poisoning value then
                by active learning value

            sampling_title (string): title used for the sampling axis

            poisoning_title (string): title used for the poisoning axis

            title (string): title used for diagram

        Returns:
            None
    """
    xs = []  # Poisoning Points
    ys = []  # Active Learning Sampling
    zs = scores.copy()  # AUT Score

    for y in sampling:
        xs.extend(poisoning)
    xs.sort()  # Sort to match scores structure

    for x in poisoning:
        ys.extend(sampling)

    # creating figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(xs, ys, zs, color='white', edgecolors='grey', alpha=0.75)
    ax.scatter(xs, ys, zs, c='red')

    # setting title and labels
    ax.set_title(title)
    ax.set_xlabel(poisoning_title)
    ax.set_ylabel(sampling_title)
    ax.set_zlabel(y_title)

    # displaying the plot
    plt.show()
