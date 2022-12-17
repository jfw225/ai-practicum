import os
import re
import sys
import json
import numpy as np
import matplotlib.pyplot as plt


# the stats file extension
stats_ext = ".stats"

# the regular expression to parse the test info
test_info_data_re = "`((?:\w|\d)*)`"
test_info_key_re = "(\w+)="

# the regular expression to parse the test data
data_re = "(\w+)\s+=\s+((?:\d|\.)+)"

# the list of all evaluation sets
eval_sets = [
    "bsearch",
    "cmult",
    "mfilt",
    "sort",
    "vvadd",
]


def parse_data(eval_name):
    """
    Parses the statistics data.
    """

    # construct the file name
    filename = os.path.join("data", eval_name + stats_ext)
    print(filename)

    # open the file
    f = open(filename, "r")

    # initialize the data
    eval_data = []

    # initialize the test data
    test_data = {}

    # while there are lines to read
    while True:
        # read the next line
        line = f.readline()

        # if there are no more lines to read, break
        if not line:
            break

        # if the test info keys RE matches
        if re.search(test_info_key_re, line):
            # if the test data is not empty
            if test_data:
                # add the test data to the data
                eval_data.append(test_data)

            # initialize the test data
            test_data = {"data": {}}

            # parse the test info keys
            test_info_keys = re.findall(test_info_key_re, line)

            # parse the test info data
            test_info_data = re.findall(test_info_data_re, line)

            # store the test info
            test_data["info"] = dict(zip(test_info_keys, test_info_data))

        # if the data RE matches
        if re.search(data_re, line):
            # parse the data
            [(key, value)] = re.findall(data_re, line)

            # store the data
            test_data["data"][key] = float(value)

    # add the last test data to the data
    eval_data.append(test_data)

    # close the file
    f.close()

    # determine the baseline cycle count (the cycle count of the
    # baseline implementation on the ubmark benchmark type)
    baseline_cycle_count = next(
        (x["data"]["num_cycles"] for x in eval_data if
         x["info"]["benchmark"] == "ubmark" and
         x["info"]["impl"] == "base"),
        None
    )

    # determine the number of instructions for `ubmark`
    ubmark_num_inst = next(
        (x["data"]["num_inst"] for x in eval_data if
            x["info"]["benchmark"] == "ubmark" and
            x["info"]["impl"] == "base"),
        None
    )

    # determine the number of instructions for `mtbmark`
    mtbmark_num_inst = next(
        (x["data"]["num_inst"] for x in eval_data if
            x["info"]["benchmark"] == "mtbmark" and
            x["info"]["impl"] == "base"),
        None
    )

    # make a couple of changes to the data
    for test_data in eval_data:
        # get the info and data
        info = test_data["info"]
        data = test_data["data"]

        # change the `benchmark` key to `btype`
        info["btype"] = info.pop("benchmark")

        # if `num_cores` is not set
        if not info["num_cores"]:
            # if the `btype`` is `mtbmark` and the `impl` is `alt`, set it to 4.
            # otherwise, set it to 1
            info["num_cores"] = 4 if (
                info["btype"] == "mtbmark" and
                info["impl"] == "alt") else 1
        # otherwise, convert it to an integer
        else:
            info["num_cores"] = int(info["num_cores"])

        # add a key for the relative cycle count
        data["norm_num_cycles"] = data["num_cycles"] / baseline_cycle_count

        # compute the speedup from the baseline implementation
        data["speedup"] = np.round(1 / data["norm_num_cycles"], 1)

        # TODO: remove this after instruction stats are figured out
        # reset the number of instructions to the baseline implementation
        data["num_inst"] = ubmark_num_inst if info["btype"] == "ubmark" else mtbmark_num_inst

        # compute the new CPI
        data["CPI"] = np.round(data["num_cycles"] / data["num_inst"], 1)

    # return the data
    return eval_data


def get_avg_data(eval_data):
    # initialize the average data
    avg_data = []

    # compute the average of the data
    for eval_set, data in eval_data.items():
        # for each test
        for test_data in data:
            # pop the `info.test_name` key
            test_data["info"].pop("test_name")

            # find the object in `avg_data` with the same info as `test_data`
            # but ignore the `info.test_name` key
            avg_test_data = next(
                (x for x in avg_data if
                 all([x["info"][k] == v for k, v in test_data["info"].items()])),
                None
            )

            # if `avg_test_data` is `None`
            if avg_test_data is None:
                # then initialize it
                avg_test_data = {
                    "info": {"test_name": "avg", **test_data["info"]},
                    "data": {}
                }

                # add the test data to the average data
                avg_data.append(avg_test_data)

            # for each key, value in the test data
            for key, value in test_data["data"].items():
                # if the key is not in the average test data
                if key not in avg_test_data["data"]:
                    # then initialize it
                    avg_test_data["data"][key] = 0

                # add the value to the average test data
                avg_test_data["data"][key] += value / len(eval_sets)

    return avg_data


def plot_eval_data(eval_set, data, ax1=None, ax2=None, index=None, total=None):
    # generate a list of indices
    indices = np.arange(len(data)).astype(np.float64)

    # initialize the color to `None`
    color = None

    # initialize the width
    width = 0.125

    # get the test info and sort it by key
    test_info = [sorted(test_data["info"].items(), key=lambda x: x[0])
                 for test_data in data]

    # generate the labels for each index
    labels = ["\n".join(
        [f"{k}={v}" for k, v in info if k !=
         "test_name"]
    ) for info in test_info]

    # if either `ax1` or `ax2` is `None`, then initialize the subplots
    if ax1 is None or ax2 is None:
        # ensure that they are both `None`
        assert ax1 is None and ax2 is None

        # assign a color to each index
        color = [f"C{i}" for i in indices]

        # set the width to 0.8 which is the default width
        width = 0.8

        # set the x offset to zero
        x_offset = 0

        # initialize the subplots (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # set the figure title
        fig.suptitle(f"Evaluation Results for Benchmark: \"{eval_set}\"" if eval_set !=
                     "avg" else "Average Evaluation Results for All Benchmark Sets")

    # otherwise, we are plotting all of the data
    else:
        # set the x offset (centered around `total/2`)
        x_offset = (index - total / 2) * width

    # set the labels for `ax1`
    ax1.set_title("Speedup of the Program Relative to the Baseline")
    ax1.set_ylabel("Speedup")

    # set the labels for `ax2`
    ax2.set_title("CPI of the Program")
    ax2.set_ylabel("CPI")

    # determine the label for the bar
    label = eval_set if eval_set != "avg" else "Average for\nAll Benchmark Sets"

    # plot the the speedup
    rects1 = ax1.bar(
        indices + x_offset,
        [test_data["data"]["speedup"]
         for test_data in data],
        label=label,
        color=color,
        width=width
    )

    # plot the CPIs
    rects2 = ax2.bar(
        indices + x_offset,
        [test_data["data"]["CPI"] for test_data in data],
        label=label,
        color=color,
        width=width
    )

    # set the tick labels for `ax1`
    ax1.set_xticks(indices, labels)

    # set the tick labels for `ax2`
    ax2.set_xticks(indices, labels)

    # add the bar labels
    # ax1.bar_label(rects1, padding=3)
    # ax2.bar_label(rects2, padding=3)


def plot_all_data(eval_data):
    # initialize the subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # set the figure title
    fig.suptitle("Evaluation Results for All Benchmark Sets")

    # for each eval set
    for i, (eval_set, data) in enumerate(eval_data.items()):
        # plot the data
        plot_eval_data(eval_set, data, ax1, ax2, i, len(eval_data))

    # add a legend to `ax1` in the top left
    ax1.legend(loc="upper left")

    # add a legend to `ax2` in the top right
    ax2.legend(loc="upper right")

    # add dashed horizontal lines at each x tick
    ax1.grid(axis="y", linestyle="--")
    ax2.grid(axis="y", linestyle="--")

    # set to tight layout
    fig.tight_layout()


if __name__ == '__main__':
    # if the first CLI argument is `all`, then plot all the data
    PLOT_ALL = len(sys.argv) > 1 and sys.argv[1] == "all"

    # mapping from eval set to data
    eval_data = {}

    # for each eval set
    for eval_set in eval_sets:
        # parse the data
        data = parse_data(eval_set)
        # print(eval_set, json.dumps(data, indent=4), len(data))

        # store the data
        eval_data[eval_set] = data

        # plot the data
        not PLOT_ALL and plot_eval_data(eval_set, data)

    # get the average data
    avg_data = get_avg_data(eval_data)
    # print(json.dumps(avg_data, indent=4))

    # add the average data to the eval data
    eval_data["avg"] = avg_data

    # plot all the data or the average data
    plot_all_data(eval_data) if PLOT_ALL else plot_eval_data("avg", avg_data)

    # show the plot
    plt.show()
