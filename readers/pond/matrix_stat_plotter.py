import seaborn as sns
import numpy as np
import matplotlib.pylab as plt
from enum import Enum
import os


# New Plot: Variance -> at Oth time step then at 10th time step
# X Achse -> Ranks, Y Achse -> Laufzeit (relativ could be better)
# Do a sampling point really fast

# check invasion scenario and send jobs asap

from definitions import *


def read_matrix(filepath, header, footer):
    mat = []
    file = open(filepath, "r")
    region = False

    for line in file:
        if header in line:
            region = True
            continue

        if footer in line:
            region = False
            continue

        if region:
            arr = line.split(",")
            arr = [int(el) for el in arr]
            mat.append(arr)

    return np.array(mat)


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


def read_workload_at_checkpoint(filepath, rankcount, workload_model):
    max_output_num = 0
    file = open(filepath, "r")

    for line in file:
        if "Output number" in line:
            tokens = line.split(":")
            # print("B: ", tokens[1].split(","))
            onum = int(tokens[1].split(",")[0])
            if onum > max_output_num:
                max_output_num = onum

    #print(max_output_num, ", ", rankcount)
    if max_output_num == 0:
        return np.array([])

    workload_per_model = np.zeros((max_output_num, rankcount))

    file.seek(0)
    for line in file:
        if "Output number" in line:
            tokens = line.split(":")
            onum = int(tokens[1].split(",")[0])
            #print("A: ", tokens)
            #print("B: ", tokens[1].split(","))
            rnum = int(tokens[1].split(",")[1].split("-")[1])
            #print("C: ", rnum)
            tokens = tokens[2].split("|")
            #print("D: ", tokens)

            for token in tokens:
                if workload_model in token:
                    workloads = tokens[int(workload_model[0:3])-2]
                    # print("E: ", tokens)
                    desired_workload = workloads.split(",")
                    # We can have 2 formats, old format single entry, new format with (global_memory,computed)
                    if len(desired_workload) == 2:
                        desired_workload = desired_workload[1]
                        desired_workload = desired_workload[:-2]
                    else:
                        desired_workload = desired_workload[0][5:-1]
                    ftoken = float(desired_workload)
                    workload_per_model[onum-1][rnum] = ftoken

    #print(filepath, " -> ", workload_per_model.shape)
    # print(workload_per_model)
    return workload_per_model


def plot_relative_imbalance(filename, workloads, suffix):
    plt.close()

    rel_vars = []
    at = []

    j = 0
    for i in workloads:
        rel_vars.append(np.var(i) / np.mean(i))
        at.append(str(j))
        j += 1

    # print(np.array(rel_vars).shape)
    plt.bar(at, rel_vars)
    ax = plt.gca()

    ax.set_ylabel("Variance / Mean of Workload")
    ax.set_xlabel("Sampling points")

    plt.savefig(filename + "-relative-imbalance-" + suffix + ".pdf")

    plt.close()


def plot_relative_imbalance_line(filename, workloads, suffix):
    plt.close()

    ax = plt.gca()


def plot_workloads(filename, workloads, suffix):
    plt.close()
    ax = sns.boxplot(data=workloads.T)
    ax.set_ylabel("Cost model: " +
                  workload_model_to_name[suffix])
    ax.set_xlabel("Sampling points")

    plt.savefig(filename + "-boxplot-" + suffix + ".pdf")


def plot_heatmap(matrix, title, xlabel, ylabel, prefix, suffix):
    plt.close()
    ax = sns.heatmap(matrix, cmap="YlGnBu")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.savefig(prefix + "-heatmap" + suffix + ".pdf")
    plt.close()


def generate_plots():
    for basepath in basepaths:
        all_dirs = get_dirs(basepath)
        for job in jobtypes:
            for nodecount in nodecounts:
                actors = []
                prefix = "pond-" + job + "-" + str(nodecount) + "-"
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
                        workloads = read_workload_at_checkpoint(
                            basepath + "/" + dirname + "/" + filename, int(nodecount)*28, "24")
                        if workloads.size != 0:
                            plot_workloads(filename, workloads, "24")
                            plot_relative_imbalance(filename, workloads, "24")


def generate_line_plots():
    for basepath in basepaths:
        all_dirs = get_dirs(basepath)

        for nodecount in nodecounts:

            plt.close()
            plt.figure()

            ax = plt.gca()
            ax.set_ylabel("Variance / Mean of Workload")
            ax.set_xlabel("Sampling points")

            max_x = 0

            for job in jobtypes:
                actors = []
                prefix = "pond-" + job + "-" + str(nodecount) + "-"
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
                        workloads = read_workload_at_checkpoint(
                            basepath + "/" + dirname + "/" + filename, int(nodecount)*28, "23")
                        if workloads.size != 0:
                            rel_vars = []
                            at = []

                            j = 0
                            for i in workloads:
                                rel_vars.append(np.var(i) / np.mean(i))
                                at.append(str(j))
                                j += 1

                            if job != "static" and job != "sw-static":
                                if j > max_x:
                                    max_x = j
                                    #print(max_x, " | ", job)

                            # print(np.array(rel_vars).shape)
                            plt.plot(at, rel_vars, label=job)

            plt.legend()
            plt.xlim((-0.2, max_x+2.2))
            ax.set_yscale('log')
            plt.savefig(basepath + "-" + str(nodecount) +
                        "-relative-imbalance-line" + ".pdf")


def generate_two_sample_plots():
    for basepath in basepaths:
        all_dirs = get_dirs(basepath)

        # 1 plot per nodecount to compare strategies
        for nodecount in nodecounts:
            # Fresh start new plot
            plt.close()
            plt.figure()

            # 2 subplots for sample at time 0 and sample at time 'much later'
            fig, axes = plt.subplots(len(jobtypes), 2)
            plt.gcf().tight_layout()
            #plt.subplots_adjust(left=0.5, right=0.5, top=0.5, bottom=0.5)
            #plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            fig.text(-0.04, 0.5, 'Workload per rank',
                     va='center', rotation='vertical')
            # Set labels

            y_maxes = np.zeros(len(jobtypes)*2)

            for i in range(len(jobtypes)):
                #axes[i, 0].set_ylabel("Workload per rank")
                if i == len(jobtypes)-1:
                    axes[i, 0].set_xlabel(
                        job_to_label[jobtypes[i]] + "\n \n  Sample 1")
                else:
                    axes[i, 0].set_xlabel(job_to_label[jobtypes[i]])

                #axes[i, 0].set_xlim([-2, int(nodecount)*28+2])
                #axes[i, 1].set_xlim([-2, int(nodecount)*28+2])
                #axes[i, 0].set_yscale("log")
                #axes[i, 1].set_yscale("log")
            # plt.ylabel("Workload per rank")

            max_samples = np.zeros(len(jobtypes))
            all_samples = []
            rankids = np.array(list(range(0, int(nodecount)*28)))

            i = 0
            for job in jobtypes:
                prefix = "pond-" + job + "-" + str(nodecount) + "-"
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
                        workloads = read_workload_at_checkpoint(
                            basepath + "/" + dirname + "/" + filename, int(nodecount)*28, "24")
                        if workloads.size != 0:
                            rel_vars = []
                            at = []

                            j = 0
                            for wl in workloads:
                                rel_vars.append(np.var(wl) / np.mean(wl))
                                at.append(str(j))
                                j += 1

                            #if job != "static" and job != "sw-static":
                            if True:
                                if j > max_samples[i]:
                                    max_samples[i] = j
                                    #print(max_x, " | ", job)

                            # print(np.array(rel_vars).shape)
                            # plt.plot(at, rel_vars, label=job)
                        all_samples.append(workloads)
                
                i += 1
            i = 0

            max_sample_end = np.max(max_samples)
            min_sample_end = np.min(max_samples)
            right_sample = max(int(min_sample_end * 0.7), 1)
            print(jobtypes)
            print(nodecount, ": ", max_samples, " and ", min_sample_end, ", right sample: ", right_sample)
            y_max = 0.0

            print(len(all_samples), ", ", len(jobtypes))

            i = 0
            for job in jobtypes:
                # Gather maximal y values for y limit
                y_maxes[2*i] = -np.sort(-all_samples[i][0, :])[0]
                y_maxes[2*i+1] = -np.sort(-all_samples[i][right_sample, :])[0]
                i += 1
            i = 0

            print("Y-maxes: ", y_maxes)
            y_max_inds = np.amax(y_maxes)
            # print(y_max_inds)
            y_max = y_max_inds
            print("Y-max: ", y_max)

            i = 0
            for job in jobtypes:
                axes[i, 0].set_ylim(bottom=0, top=y_max)
                axes[i, 1].set_ylim(bottom=0, top=y_max)
                i += 1
            i = 0

            i = 0
            for job in jobtypes:
                #print("Sample shape: ", all_samples[i].shape)
                #print("Sample left: ", all_samples[i][0, :])
                #print("Sample right: ", all_samples[i][right_sample, :])
                sleft = -np.sort(-all_samples[i][0, :])
                sright = -np.sort(-all_samples[i][right_sample, :])
                #print("Sample left (desc): ", sleft)
                #print("Sample right (desc): ", sright)

                axes[i, 0].bar(rankids,
                               sleft,
                               color=color_dict[job_to_label[job]])
                axes[i, 1].bar(rankids,
                               sright,
                               color=color_dict[job_to_label[job]])

                if i == len(jobtypes)-1:
                    axes[i, 1].set_xlabel(
                        job_to_label[jobtypes[i]] + "\n \n Sample " + str(right_sample))
                else:
                    axes[i, 1].set_xlabel(
                        job_to_label[jobtypes[i]])

                i += 1

            plt.savefig(basepath + "-" + str(nodecount) +
                        "-two-sample" + ".pdf", bbox_inches='tight')


# generate_plots()
# generate_line_plots()
generate_two_sample_plots()

"""
comm_mat = read_matrix("exmaple", "COMM_MATRIX BEGIN", "COMM_MATRIX END")
mig_mat = read_matrix("exmaple", "MIGRATION_MATRIX BEGIN",
                      "MIGRATION_MATRIX END")


plot_heatmap(comm_mat, "Communication Matrix", "UPC++ Ranks",
             "UPC++ Ranks", "communication", "")
plot_heatmap(comm_mat, "Migration Matrix", "UPC++ Ranks",
             "UPC++ Ranks", "migraiton", "")

workloads = read_workload_at_checkpoint("output", 5, "27")
plot_workloads(workloads, "27")
workloads = read_workload_at_checkpoint("output", 5, "24")
plot_workloads(workloads, "24")
workloads = read_workload_at_checkpoint("output", 5, "23")
plot_workloads(workloads, "23")
"""
