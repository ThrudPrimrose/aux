import os
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np

jobtypes = ["static", "global-busy",
            "global-random", "local-busy", "local-random"]

nodecounts = ["4", "8", "16", "24", "32"]

basepaths = ["scenario3-huge-lazy-strongscaling", "huge-lazy-strongscaling"]

suffix = ".out"

markers = [".", "o", "^", "*", "+", "x", "D", "|"]

linestyles = [":", "-.", "--", "-"]


def get_files(path):
    onlyfiles = [f for f in os.listdir(
        path) if os.path.isfile(os.path.join(path, f))]
    return onlyfiles


def get_dirs(path):
    onlydirs = [f for f in os.listdir(
        path) if os.path.isdir(os.path.join(path, f))]
    return onlydirs


def filter_prefix(prefix, vec):
    matches = [f for f in vec if f.startswith(prefix)]
    # print(matches)
    return matches


def filter_suffix(suffix, vec):
    matches = [f for f in vec if f.endswith(suffix)]
    # print(matches)
    return matches


def check_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


comp_limits = dict()


def get_computation_lines():
    for basepath in basepaths:

        comp_limits[basepath] = []

        all_dirs = get_dirs(basepath)

        for job in ["static"]:

            for nodecount in nodecounts:
                prefix = "pond-" + job + "-" + nodecount + "-"
                # print(prefix)

                matching_dirs = filter_prefix(prefix, all_dirs)
                # print(matching_dirs)

                for dirname in matching_dirs:
                    # print(name)
                    outfiles = get_files(basepath + "/" + dirname)
                    outfiles = filter_suffix(suffix, outfiles)
                    outfiles.sort()
                    computation_limit = 0.0
                    # print(outfiles)
                    if len(outfiles) > 0:
                        filename = outfiles[-1]

                        with open(basepath + "/" + dirname + "/" + filename, 'r') as output_file:
                            for line in output_file:
                                if "(5)" in line:
                                    tokens = line.split("|")

                                    for token in tokens:
                                        if "(5)" in token and not "TIME_ACT" in token:
                                            token = token[4:]
                                            # print("act_time: ", token)
                                            computation_limit += float(token)

                            # print(computation_limit)
                            computation_limit = computation_limit / \
                                (28.0 * float(nodecount))
                            comp_limits[basepath].append(computation_limit)
                    else:
                        comp_limits[basepath].append(0.0)


def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return np.array([float('nan') if x == 0 else x for x in values])


for basepath in basepaths:
    datas = []
    flops = []

    all_dirs = get_dirs(basepath)

    get_computation_lines()

    for job in jobtypes:
        data = []
        flop = []

        for nodecount in nodecounts:
            prefix = "pond-" + job + "-" + nodecount + "-"
            # print(prefix)

            matching_dirs = filter_prefix(prefix, all_dirs)
            # print(matching_dirs)

            for dirname in matching_dirs:
                # print(name)
                outfiles = get_files(basepath + "/" + dirname)
                outfiles = filter_suffix(suffix, outfiles)
                outfiles.sort()
                # print(outfiles)
                if len(outfiles) > 0:
                    filename = outfiles[-1]
                    found = False
                    foundflops = False
                    with open(basepath + "/" + dirname + "/" + filename, 'r') as output_file:
                        for line in output_file:
                            if "time-to-solution:" in line and not found:
                                tokens = line.split(":")

                                for token in tokens:
                                    if check_float(token):
                                        data.append(float(token))
                                        found = True

                                if not found:
                                    after = line.split(
                                        "time-to-solution:", 1)[1]
                                    floatchars = ""
                                    for ch in after:
                                        if ch.isdigit() or ch == '.':
                                            floatchars += ch
                                        if not ch.isdigit() and ch != '.':
                                            break
                                    data.append(float(floatchars))
                                    found = True
                            if "process 0 - =>" in line and "FlOps/s" in line:
                                l = line.split("=>", 1)[1]
                                spll = l.split(" ")
                                for token in spll:
                                    if check_float(token):
                                        flop.append(float(token))
                                        foundflops = True

                        if not found:
                            data.append(0.0)
                            found = True
                        if not foundflops:
                            flop.append(0.0)
                            foundflops = True
                else:
                    data.append(0.0)
                    flop.append(0.0)

        # datas.append(data)
        print(job + ": ", data)
        print(job + ": ", flop)
        datas.append(np.array(data))
        flops.append(np.array(flop))

    # print(datas)
    # print(flops)

    # some markers for line styles:
    # taken from: https://pydatascience.org/2017/11/24/plot-multiple-lines-in-one-chart-with-different-style-python-matplotlib/
    # ax.plot(x,x,c='b',marker="^",ls='--',label='Greedy',fillstyle='none')
    # ax.plot(x,x+1,c='g',marker=(8,2,0),ls='--',label='Greedy Heuristic')
    # ax.plot(x,(x+1)**2,c='k',ls='-',label='Random')
    # ax.plot(x,(x-1)**2,c='r',marker="v",ls='-',label='GMC')
    # ax.plot(x,x**2-1,c='m',marker="o",ls='--',label='KSTW',fillstyle='none')
    # ax.plot(x,x-1,c='k',marker="+",ls=':',label='DGYC')

    with matplotlib.backends.backend_pdf.PdfPages(basepath + ".pdf") as pdf1:
        plt.figure()
        plt.title("Time-to-solution:")

        for i in range(len(datas)):
            data = datas[i]
            label = jobtypes[i]
            plt.plot(nodecounts, zero_to_nan(data), marker=markers[i % len(
                markers)], ls=linestyles[i % len(linestyles)], label=label,
                linewidth=0.75, markersize=2.5)
            #plt.scatter(nodecounts, zero_to_nan(data), s=5, label=label)

        # print("uwu")
        print(comp_limits[basepath])
        plt.plot(nodecounts, comp_limits[basepath],
                 label="Time spent in act", linewidth=0.75)

        plt.xlabel("Node counts (28 ranks per node)")
        plt.ylabel("Time to solution")

        plt.legend()
        pdf1.savefig()

    with matplotlib.backends.backend_pdf.PdfPages(basepath + "-flops.pdf") as pdf2:
        plt.figure()
        plt.title("Flop/s:")

        for i in range(len(flops)):
            data = flops[i]
            data = np.array(data)
            # print("plot:", data)
            label = jobtypes[i]
            plt.plot(nodecounts, zero_to_nan(data), marker=markers[i % len(
                markers)], ls=linestyles[i % len(linestyles)], label=label,
                linewidth=0.75, markersize=2.5)

            plt.xlabel("Node counts (28 ranks per node)")
            plt.ylabel("Flop/s")

            #plt.scatter(nodecounts, zero_to_nan(data), s=5, label=label)

        plt.legend()
        pdf2.savefig()

        # Prediktiv von dem Rank Stehelen der am meisten gearbeitet hat
        # Global von dem meist gearbeiteten Rank zu stehlen
        # Global zufallig
        # check correctness of output
