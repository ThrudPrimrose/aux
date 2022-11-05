from operator import truediv
import os
from re import A
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
import pandas as pd

from definitions import *

gathered_avg_act_time = False
loc_str = "upper left"

fo = open('percentages', 'a')



# nodecounts = ["1", "2", "4", "8", "16", "32"]
# nodecountsint = [1, 2, 4, 8, 16,  32]
# xticks = nodecounts
# nodecountlabels = ["1", "2", "4", "8", "16", "32"]

# basepaths = ["sc3-strongscaling", "sc3-strongscaling-newcd",
#             "sc3-strongscaling-newcd-gawaylim", "sc3-strongscaling-nocd"]

suffix = ".out"

chars_to_skip = 0

base_time_per_strategy = []


actor_vars = []
actor_means = []

static_no_slowdown = []


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
    global gathered_avg_act_time
    if not gathered_avg_act_time:
        gathered_avg_act_time = True
        for basepath in basepaths:

            comp_limits[basepath] = []

            all_dirs = get_dirs(basepath)
            global static_job_name
            for job in [static_job_name]:

                for nodecount in nodecounts:
                    prefix = "pond-" + job + "-" + str(nodecount) + "-"
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
                                    if "(5)" in line and "Output number" not in line:
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


def get_pure_static():
    bp = "sc3-staticworkload"

    all_dirs = get_dirs(bp)
    for job in [static_job_name]:
        data = []
        flop = []

        for nodecount in nodecounts:
            prefix = "pond-" + job + "-" + str(nodecount) + "-"
            # print(prefix)

            matching_dirs = filter_prefix(prefix, all_dirs)
            # print(matching_dirs)

            for dirname in matching_dirs:
                # print(name)
                outfiles = get_files(bp + "/" + dirname)
                outfiles = filter_suffix(suffix, outfiles)
                outfiles.sort()
                # print(outfiles)
                if len(outfiles) > 0:
                    filename = outfiles[-1]
                    found = False
                    foundflops = False
                    with open(bp + "/" + dirname + "/" + filename, 'r') as output_file:
                        for line in output_file:
                            if "time-to-solution:" in line and not found:
                                tokens = line.split(":")

                                for token in tokens:
                                    if check_float(token):
                                        static_no_slowdown.append(float(token))
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
                                    static_no_slowdown.append(
                                        float(floatchars))
                                    found = True

                        if not found:
                            output_file.seek(0)
                            for line in output_file:
                                if "Rank-0:" in line and not "Output number" in line:
                                    toks = line.split("|")
                                    prt = toks[12]
                                    runtime = float(prt[5:])
                                    static_no_slowdown.append(runtime)
                                    found = True
                                    break

                        if not found:
                            static_no_slowdown.append(0.0)
                            found = True
                else:
                    static_no_slowdown.append(0.0)

        # datas.append(data)
        print(bp, " | ", job + ": ", static_no_slowdown)


get_pure_static()


def actor_var_and_mean():
    global actor_means
    global actor_vars

    for basepath in basepaths:
        all_dirs = get_dirs(basepath)
        for nodecount in nodecounts:
            actors = []
            prefix = "pond-" + static_job_name + "-" + str(nodecount) + "-"
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
                            if "Rank-" in line and "(4)" in line and not "Output number" in line:
                                tokens = line.split("(4)")
                                tokens = tokens[1].split("|")
                                tokens = tokens[0].split(",")
                                actors.append(int(tokens[0]))
            actors_np = np.array(actors)
            print(actors_np)
            actor_means.append(np.mean(actors_np))
            actor_vars.append(np.var(actors_np))


actor_var_and_mean()
print(actor_means)
print(actor_vars)


def get_successful_steal_percentages():
    global gathered_avg_act_time
    global nodecounts
    global basepaths
    global jobtypes

    percentages_all = []
    total_steals = []
    total_attempts = []

    for basepath in basepaths:
        all_dirs = get_dirs(basepath)

        for job in jobtypes:
            percentages = []
            ts = []
            ta = []

            for nodecount in nodecounts:
                tries_total = 0
                success_total = 0

                prefix = "pond-" + job + "-" + str(nodecount) + "-"
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
                                if "Output number" in line:
                                    tokens = line.split(":")
                                    tokens = tokens[2].split("|")

                                    for token in tokens:
                                        if "(1)" in token and not "STOLE_COUNT" in token:
                                            # print(token)
                                            token = token[4:]
                                            # print("stole_count: ", token)
                                            success_total += int(token)

                                    for token in tokens:
                                        if "(2)" in token and not "STOLE_TRIES" in token:
                                            # print(token)
                                            token = token[4:]
                                            # print("try_count: ", token)
                                            tries_total += int(token)
                    else:
                        success_total += 0
                        tries_total += 0
                if (tries_total == 0):
                    tries_total = 1
                percentages.append(success_total / tries_total)
                ts.append(success_total)
                ta.append(tries_total)
            print(job, ": ", percentages)
            percentages_all.append(percentages)
            total_attempts.append(ta)
            total_steals.append(ts)
    return percentages_all, total_attempts, total_steals


successful_steal_percentage, total_steal_attempts, total_steals = get_successful_steal_percentages()


def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return np.array([float('nan') if x == 0.0 else x for x in values])


speedups = []


for basepath in basepaths:
    datas = []
    flops = []

    all_dirs = get_dirs(basepath)

    get_computation_lines()

    for job in jobtypes:
        data = []
        flop = []

        for nodecount in nodecounts:
            prefix = "pond-" + job + "-" + str(nodecount) + "-"
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
                            output_file.seek(0)
                            for line in output_file:
                                if "Rank-0:" in line and not "Output number" in line:
                                    toks = line.split("|")
                                    prt = toks[12]
                                    runtime = float(prt[5:])
                                    data.append(runtime)
                                    found = True
                                    break

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
        print(basepath, " | ", job + ": ", data)
        print(basepath, " | ", job + ": ", flop)
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

    static_baseline = datas[0]

    with matplotlib.backends.backend_pdf.PdfPages(basepath + "-relative-speedup.pdf") as pdf1:
        plt.figure()
        # plt.title("Speedup:")
        plt.subplots_adjust(right=0.72)

        for i in range(len(datas)):
            label = jobtypes[i]
            # if (label == "sw-offload-local-single"):
            #    datas[i][5] = 135.0
            data = np.divide(static_baseline, datas[i])
            speedups.append(data)
            # if len(data) == 0:
            #    data = [0.0] * len(nodecounts)

            # print(data)
            # print(nodecounts)
            # if (label[3:] == "static"):
            #    label = "sw-baseline"

            plt.plot(nodecounts, zero_to_nan(data), marker=markers[i % len(
                markers)], ls=linestyles[i % len(linestyles)], label=job_to_label[label],
                linewidth=0.75, markersize=2.5, color=color_dict[job_to_label[label]])
            # if (label == "sw-offload-local-single"):
            #    ldata = [static_baseline[5] / 135.0]
            #    v = ["32"]
            #    plt.plot(v, zero_to_nan(ldata), marker="x", ls=linestyles[i % len(linestyles)], label=job_to_label[label],
            #             linewidth=0.75, markersize=4.5, color="red")
            # plt.scatter(nodecounts, zero_to_nan(data), s=5, label=label)

            # print("uwu")

        if "slowdown" in basepaths[0]:
            npnoslowdownstatic = np.array(static_no_slowdown)
            ldata = np.divide(static_baseline,
                              npnoslowdownstatic)
            print("ldata: ", ldata)
            label = "Ideal LB"
            plt.xticks(nodecounts, labels=nodecountlabels)
            plt.plot(nodecounts, zero_to_nan(ldata), ls="--", label=job_to_label[label],
                linewidth=0.75, markersize=2.5, color=color_dict[job_to_label[label]])

        print(comp_limits[basepath])
        data = np.divide(static_baseline, comp_limits[basepath])

        if not "slowdown" in basepaths[0]:
            plt.plot(nodecounts, data,  # ls=":",
                     label="Ideal LB", ls="--", linewidth=0.75, color=color_dict["Ideal LB"])

        plt.xlabel("Node count (28 ranks per node)")
        plt.ylabel("Speedup")

        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.grid(True, which="both", axis="y", alpha=0.7,
                 color="lightgray", linewidth=0.2)
        plt.xticks(xticks)
        pdf1.savefig()

    with matplotlib.backends.backend_pdf.PdfPages(basepath + "-speedup.pdf") as pdf1:
        plt.figure()
        # plt.title("Speedup:")
        plt.subplots_adjust(right=0.72)

        base_times = []
        for i in range(len(datas)):
            base_times.append(datas[i][0])

        speedups = []
        relativespeedups = []
        for i in range(len(datas)):
            data = np.divide([base_times[0]]*len(datas[0]), datas[i])
            label = jobtypes[i]
            speedups.append(data)
            reldata = np.divide(static_baseline, datas[i])
            print("rrr: ", label, ": ", reldata)
            relativespeedups.append(reldata)
            # if len(data) == 0:
            #    data = [0.0] * len(nodecounts)

            # print(data)
            # print(nodecounts)
            # if (label[3:] == "static"):
            #    label = "sw-baseline"

            # plt.yscale("log")
            # plt.xscale("log", base=2)
            # plt.xticks(nodecounts, labels=nodecountlabels)
            plt.plot(nodecounts, zero_to_nan(data), marker=markers[i % len(
                markers)], ls=linestyles[i % len(linestyles)], label=job_to_label[label],
                linewidth=0.75, markersize=2.5, color=color_dict[job_to_label[label]])
            # plt.scatter(nodecounts, zero_to_nan(data), s=5, label=label)

        if "slowdown" in basepaths[0]:
            npnoslowdownstatic = np.array(static_no_slowdown)
            for i in range(len(npnoslowdownstatic)):
                if npnoslowdownstatic[i] < 300:
                    npnoslowdownstatic[i] = npnoslowdownstatic[i] * 23/21
                else:
                    loss_time = 300 * 2/21
                    npnoslowdownstatic[i] = npnoslowdownstatic[i] + loss_time

            ldata = np.divide([base_times[0]]*len(datas[0]),
                              npnoslowdownstatic)
            label = "Ideal LB"
            # plt.yscale("log")
            # plt.xscale("log", base=2)
            # plt.xticks(nodecounts, labels=nodecountlabels)
            plt.plot(nodecounts, zero_to_nan(ldata), ls="--", label=label,
                linewidth=0.75, markersize=2.5, color=color_dict[label])

        if "lazyactivation" in basepaths[0]:

            label = "Ideal LB"
            ldata = np.divide([base_times[0]]*len(datas[0]),
                              comp_limits[basepath])
            # plt.yscale("log")
            # plt.xscale("log", base=2)
            # plt.xticks(nodecounts, labels=nodecountlabels)
            plt.plot(nodecounts, zero_to_nan(ldata), ls="--", label=label,
                linewidth=0.75, markersize=2.5, color=color_dict[label])

        # print("uwu")
        print("comp limits: ", comp_limits[basepath])

        speedups = np.array(speedups)
        relativespeedups = np.array(relativespeedups)

        datasnp = np.array(datas)
        # for j in range(len(datas)):
        #    datasnp[i] = np.divide(static_baseline, datasnp[i])

        # print(datasnp)
        print("sp1: ", speedups)
        print("rs1: ", relativespeedups)
        minspeedups = speedups

        speedups = np.nan_to_num(
            speedups, nan=1.0, neginf=0.999999, posinf=0.9999999)
        print("sp2: ", speedups)

        relativespeedups = np.nan_to_num(
            relativespeedups, nan=1.0, neginf=0.999999, posinf=0.999999)
        print("rs2: ", relativespeedups)

        relativeminspeedups = np.nan_to_num(
            relativespeedups, nan=1.999999, neginf=1.999999, posinf=1.999999)
        print("rs2: ", relativespeedups)

        colmaxs = speedups.max(axis=0)
        colmins = minspeedups.min(axis=0)
        relcolmaxs = relativespeedups.max(axis=0)
        relcolmins = relativeminspeedups.min(axis=0)
        # print(speedups)
        print(colmaxs)
        print(colmins)

        ax = plt.gca()
        for i in range(len(nodecounts)):
            colmax = colmaxs[i]
            relcolmax = relcolmaxs[i]
            dd = speedups[:, i]
            colmax = np.max(dd)
            indexmax = np.where(dd == colmax)
            print("ddmax: ", dd, " colmax: ",
                  colmax, "indexmax: ", indexmax[0])
            colmaxformatted = res = "{:.2f}".format(
                (relcolmax-1.0)*100.0) + "%"

            if len(indexmax[0]) == 0:
                indexmax = [(0, )]
                colmaxformatted = colminformatted = res = "{:.2f}".format(
                    (relativespeedups[0][i]-1.0)*100.0) + "%"
            if colmax == 1.0:
                indexmax = [(0, )]
                colminformatted = colminformatted = res = "{:.2f}".format(
                    (relativespeedups[0][i]-1.0)*100.0) + "%"
            ax.text(nodecounts[i], dd[indexmax[0][0]] + 1.5, s=colmaxformatted, color=color_dict[job_to_label[jobtypes[indexmax[0][0]]]], fontsize="xx-small",
                    bbox=dict(facecolor='none', edgecolor=color_dict[job_to_label[jobtypes[indexmax[0][0]]]], boxstyle='square', pad=0.25, linewidth=0.25))
            # plt.annotate(colmaxformatted, )
            colmin = colmins[i]
            indexmin = np.where(dd == colmin)
            print(indexmin[0])
            relcolmin = relcolmins[i]
            colminformatted = res = "{:.2f}".format((relcolmin-1.0)*100) + "%"
            print("ddmin: ", dd)
            print(indexmin[0])
            if len(indexmin[0]) == 0:
                indexmin = [(0, )]
                colminformatted = colminformatted = res = "{:.2f}".format(
                    (relativeminspeedups[0][i]-1.0)*100) + "%"
            if colmin == 1.0:
                indexmin = [(0, )]
                colminformatted = colminformatted = res = "{:.2f}".format(
                    (relativeminspeedups[0][i]-1.0)*100) + "%"
            ax.text(nodecounts[i], dd[indexmin[0][0]] - 1.5, s=colminformatted, color=color_dict[job_to_label[jobtypes[indexmin[0][0]]]], fontsize="xx-small",
                    bbox=dict(facecolor='none', edgecolor=color_dict[job_to_label[jobtypes[indexmin[0][0]]]], boxstyle='square', pad=0.25, linewidth=0.25))

        plt.xlabel("Node count (28 ranks per node)")
        plt.ylabel("Speedup")

        plt.tick_params(which="minor")
        # plt.yticks([2**i for i in range(5)])
        plt.ylim((-2, yyy))

        plt.legend(bbox_to_anchor=(1.44, 1), borderaxespad=0)
        plt.grid(True, which="both", axis="y", alpha=0.7,
                 color="lightgray", linewidth=0.2)
        # plt.xticks(xticks)
        # plt.yticks([1, 5, 10, 50], ["1", "5", "10", "50"])
        pdf1.savefig()

    with matplotlib.backends.backend_pdf.PdfPages(basepath + ".pdf") as pdf1:
        plt.figure()
        plt.subplots_adjust(right=0.72)

        for i in range(len(datas)):
            data = datas[i]
            label = jobtypes[i]

            if len(data) == 0:
                data = [0.0] * len(nodecounts)

            print(label, ": ", data)
            # print(nodecounts)

            plt.plot(nodecounts, zero_to_nan(data), marker=markers[i % len(
                markers)], ls=linestyles[i % len(linestyles)], label=job_to_label[label],
                linewidth=0.75, markersize=2.5, color=color_dict[job_to_label[label]])
            # plt.scatter(nodecounts, zero_to_nan(data), s=5, label=label)

        # print("uwu")
        print(comp_limits[basepath])
        plt.plot(nodecounts, comp_limits[basepath],
                 label="Time spent in act", linewidth=0.75)

        plt.xlabel("Node count (28 ranks per node)")
        plt.ylabel("Time to solution")

        plt.legend(bbox_to_anchor=(1.45, 1), borderaxespad=0)
        plt.grid(True, which="both", axis="y", alpha=0.7,
                 color="lightgray", linewidth=0.2)
        plt.xticks(xticks)
        pdf1.savefig()

    with matplotlib.backends.backend_pdf.PdfPages(basepath + "-flops.pdf") as pdf2:
        plt.figure()
        # plt.title("Flop/s:")
        plt.subplots_adjust(right=0.72)

        for i in range(len(flops)):
            data = flops[i]
            data = np.array(data)
            if len(data) == 0:
                data = [0.0] * len(nodecounts)
            label = jobtypes[i]
            plt.plot(nodecounts, zero_to_nan(data), marker=markers[i % len(
                markers)], ls=linestyles[i % len(linestyles)], label=job_to_label[label],
                linewidth=0.75, markersize=2.5, color=color_dict[job_to_label[label]])

            plt.xlabel("Node count (28 ranks per node)")
            plt.ylabel("Flop/s")

            # plt.scatter(nodecounts, zero_to_nan(data), s=5, label=label)

        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.grid(True, which="both", axis="y", alpha=0.7,
                 color="lightgray", linewidth=0.2)
        plt.xticks(xticks)
        pdf2.savefig()

        # Prediktiv von dem Rank Stehelen der am meisten gearbeitet hat
        # Global von dem meist gearbeiteten Rank zu stehlen
        # Global zufallig
        # check correctness of output

    with matplotlib.backends.backend_pdf.PdfPages(basepath + "-steal-percentage.pdf") as pdf2:
        plt.figure()
        plt.subplots_adjust(right=0.72)
        for i in range(len(successful_steal_percentage)):
            data = successful_steal_percentage[i]
            data = np.array(data)

            speedupdata = np.divide(static_baseline, datas[i])
            for j in range(len(speedupdata)):
                if str(speedupdata[j]) != "inf" and str(speedupdata[j]) != "1.0":
                    fo.write(str(speedupdata[j]) + "," + str(data[j]) + "\n")

            if len(data) == 0:
                data = [0.0] * len(nodecounts)
            label = jobtypes[i]
            if job_to_label[label] != "offload-local-single":
                plt.plot(nodecounts, zero_to_nan(data), marker=markers[i % len(
                    markers)], ls=linestyles[i % len(linestyles)], label=job_to_label[label],
                    linewidth=0.75, markersize=2.5, color=color_dict[job_to_label[label]])

            plt.xlabel("Node count (28 ranks per node)")
            plt.ylabel("Percentage of succesful (steal|offloading) attempts")

            # plt.scatter(nodecounts, zero_to_nan(data), s=5, label=label)

        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.grid(True, which="both", axis="y", alpha=0.7,
                 color="lightgray", linewidth=0.2)
        plt.xticks(xticks)
        pdf2.savefig()


fig, ax = plt.subplots()
width = 0.2


d = dict()
for i in range(len(successful_steal_percentage)):
    if job_to_label[jobtypes[i]] != "Offload":
        d[job_to_label[jobtypes[i]]] = successful_steal_percentage[i]

df = pd.DataFrame(data=d, index=nodecounts)

ax = df.plot(kind='bar', rot=0, xlabel="Node count (28 ranks per node)",
             ylabel="Percentage of succesful migration attempts", color=color_dict)
print(df)

"""
for i in range(len(successful_steal_percentage)):
	data = successful_steal_percentage[i]
	data = np.array(data)
	if len(data) == 0:
		data = [0.0] * len(nodecounts)
	label = jobtypes[i]
	ax.bar(nodecounts, zero_to_nan(data), label=label)

	ax.set_xlabel("Node count (28 ranks per node)")
	ax.set_ylabel(
		"Percentage of succesful (steal|offloading) attempts")

	# plt.scatter(nodecounts, zero_to_nan(data), s=5, label=label)


"""
ax.legend(loc="upper right")
# ax.savefig()
# plt.xticks(xticks)
plt.savefig(basepath + "-steal-percentage-bb.pdf")


# fig = plt.figure(1)
# ig.subplots_adjust(bottom=0.2)
d = dict()
for i in range(len(datas)):
    d[job_to_label[jobtypes[i]]] = np.divide(static_baseline, datas[i])

df = pd.DataFrame(data=d, index=nodecounts)

ax = df.plot(kind='bar', rot=0, xlabel="Node count (28 ranks per node)",
             ylabel="Speedup", color=color_dict)

"""
for i in range(len(successful_steal_percentage)):
	data = successful_steal_percentage[i]
	data = np.array(data)
	if len(data) == 0:
		data = [0.0] * len(nodecounts)
	label = jobtypes[i]
	ax.bar(nodecounts, zero_to_nan(data), label=label)

	ax.set_xlabel("Node count (28 ranks per node)")
	ax.set_ylabel(
		"Percentage of succesful (steal|offloading) attempts")

	# plt.scatter(nodecounts, zero_to_nan(data), s=5, label=label)


"""

for basepath in basepaths:
    with matplotlib.backends.backend_pdf.PdfPages(basepath + "-actor-counts.pdf") as pdf11:
        plt.figure()
        fig, ax = plt.subplots(2)
        # plt.title("Speedup:")
        plt.subplots_adjust(right=0.72)
        # plt.xticks(xticks)
        # x[0].set_xticks(xticks)
        ax[0].bar(nodecounts, actor_means, yerr=actor_vars)

        plt.xlabel("Node count (28 ranks per node)")
        ax[0].set_ylabel("Actor count per rank")

        # ax.set_tick_params(which="minor")
        # plt.yticks([2**i for i in range(5)])
        # plt.ylim((0, 100 + 10))

        # plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        # plt.grid(True, which="both", axis="y", alpha=0.7,
        #         color="lightgray", linewidth=0.2)
        d = dict()
        for i in range(len(total_steals)):
            if jobtypes[i] != "offload-local-single" and jobtypes[i] != "sw-offload-local-single":
                d[job_to_label[jobtypes[i]]] = successful_steal_percentage[i]

        df = pd.DataFrame(data=d, index=nodecounts)

        df.plot(ax=ax[1], kind='bar', rot=0, xlabel="Node count (28 ranks per node)",
                ylabel="% of successful steal attempts", color=color_dict)
        ax[1].legend(bbox_to_anchor=(1.435, 1), borderaxespad=0)
        # ax[1].set_ylabel(ylabel="Total migrations", labelpad=1332325)
        print("Steals:\n", df)

        # plt.show()
        plt.tight_layout()
        pdf11.savefig()

d = dict()
for i in range(len(total_steals)):
    d[job_to_label[jobtypes[i]]] = total_steals[i]

df = pd.DataFrame(data=d, index=nodecounts)
plt.tight_layout()
plt.autoscale()

ax = df.plot(kind='bar', rot=0, xlabel="Node count (28 ranks per node)",
             ylabel="Total migrations", color=color_dict)
print("Steals:\n", df)

ax.legend(loc="upper right")
# ax.savefig()

plt.savefig(basepath + "-ts.pdf", bbox_inches="tight")


d = dict()
for i in range(len(total_steal_attempts)):
    d[job_to_label[jobtypes[i]]] = total_steal_attempts[i]

df = pd.DataFrame(data=d, index=nodecounts)

ax = df.plot(kind='bar', rot=0, xlabel="Node count (28 ranks per node)",
             ylabel="Total migration attempts", color=color_dict)
print("Attempts:\n", df)

ax.legend(loc="upper right")
# ax.savefig()
# plt.xticks(xticks)
plt.savefig(basepath + "-ta.pdf")
