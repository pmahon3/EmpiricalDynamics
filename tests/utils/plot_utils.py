import os
import uuid
from typing import Union

import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter


# plotting error histogram helper
# noinspection DuplicatedCode
def plot_error_histogram(
        embedding,
        data,
        statistic,
        title,
        figure_directory,
        n_samples) -> Union[bytes, str]:
    if statistic is not None:
        # data subset to plot
        data_subset = data.xs(key=statistic, level='statistic')[embedding.observers[0].observation_name]
    else:
        data_subset = data[embedding.observers[0].observation_name]

    # plot histogram
    plt.hist(
        data_subset,
        bins=int(n_samples)
    )

    # mean
    plt.axvline(data_subset.mean(), color='k', linestyle='dashed', linewidth=1)

    # plot formatting and extra
    format_plot(title, make_formatter())

    # create image path and save
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    unique_id = uuid.uuid4()
    figure_path = os.path.join(figure_directory, title + '_' + str(unique_id) + '.png')
    plt.savefig(figure_path)
    plt.close()

    return figure_path


def make_formatter() -> ScalarFormatter:
    # format axes to scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))

    return formatter


def format_plot(title: str, formatter: ScalarFormatter) -> None:
    plt.title(title)
    plt.xlabel('value')
    plt.ylabel('count')
    plt.gca().xaxis.set_major_formatter(formatter)
