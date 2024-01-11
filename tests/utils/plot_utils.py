import os
import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter


# plotting error histogram helper
def plot_error_histogram(
        embedding,
        data,
        statistic,
        title,
        figure_path,
        n_samples) -> os.PathLike:

    # format axes to scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))

    # data subset to plot
    data_subset = data.xs(key=statistic, level='statistic')[embedding.observers[0].observation_name]

    # plot
    # histogram
    plt.hist(
        data_subset,
        bins=int(n_samples)
    )
    # mean
    plt.axvline(data_subset.mean(), color='k', linestyle='dashed', linewidth=1)
    # plot formatting and extras
    plt.title(title)
    plt.xlabel('value')
    plt.ylabel('count')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig(figure_path)
    plt.close()

    return figure_path
