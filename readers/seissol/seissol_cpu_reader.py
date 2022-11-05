import os
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
import re

configtypes = ["exp"]
jobtypes = ["cluster", "commapprox", "naive"]
nodecounts = ["1", "4", "8"]
lognodecounts = [1,2,3]

ranklist = [4,8]


def communication_stats(rankcount, name, jobtype):
    sentmsg = [0] * rankcount
    sentsize = [0.0] * rankcount
    ranklist = range(rankcount)

    with open(name, 'r') as fp:
        for i, line in enumerate(fp):
            #if i >= beg_offset and i <= end_offset:
            for r in ranklist:
                if "Rank " + str(r) + " has sent: " in line and "messages" in line:
                    # find all integers, first one is rank number so take the second one
                    sentmsg[r] = [int(s)
                                      for s in line.split() if s.isdigit()][1]
                if "Rank " + str(r) + " has sent: " in line and "bytes" in line:
                    # there should be only one floating point number in the lane so we will take it
                    # print(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line))
                    sentsize[r] = float(re.findall(
                        r"[-+]?(?:\d*\.\d+|\d+)", line)[1])

    sentmsg = np.array(sentmsg)
    sentsize = np.array(sentsize)

    msg_var = np.var(sentmsg)
    # print(sentsize)
    byte_var = np.var(sentsize)

    msg_mean = np.mean(sentmsg)
    byte_mean = np.mean(sentsize)

    print(jobtype, "(", rankcount ,"): average message sent: ", msg_mean,
          " and average size sent: ", byte_mean)

    print(jobtype, "(", rankcount ,"): variance of messages sent: ", msg_var,
          " and variance of bytes sent: ", byte_var)

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


basepath = "./"
suffix = ".out"
all_dirs = get_dirs(basepath)

for cfg in configtypes:
    datas = []

    for job in jobtypes:
        data = []

        prefix = job

        matching_dirs = filter_prefix(cfg + "-" + prefix, all_dirs)

        for dirname in matching_dirs:
            for nodecount in nodecounts:
                # print(dirname)
                outfiles = get_files(basepath + dirname)
                # print(outfiles)
                outfiles = filter_suffix(suffix, outfiles)
                outfiles = filter_prefix(cfg + "-" + nodecount + "-", outfiles)
                outfiles.sort()
                # print(outfiles)
                filename = outfiles[-1]
                found = False
                with open(basepath + dirname + "/" + filename, 'r') as output_file:
                    #print(basepath + dirname + "/" + filename)
                    for line in output_file:
                        if "Elapsed time (via clock_gettime):" in line:
                            tokens = line.split()
                            for token in tokens:
                                if check_float(token):
                                    if token != "9" and token != "18" and float(token) != 9.0:
                                        data.append(float(token))
                                        found = True
                    if not found:
                        data.append(0.0)
                        found = True

                if int(nodecount) > 1:
                    communication_stats(int(nodecount), basepath + dirname + "/" + filename, job)

        # datas.append(data)
        print(job + ": ", data)
        datas.append(data)

    print(datas)

    with matplotlib.backends.backend_pdf.PdfPages("seissol_" + cfg + "_cpu_times.pdf") as pdf:
        plt.figure()
        plt.title("Time-to-solution:")

        for i in range(len(datas)):
            data = datas[i]
            label = jobtypes[i]
            plt.scatter(nodecounts, data, label=label)
            #plt.ylim(bottom=1)

        plt.legend()
        pdf.savefig()

    # Prediktiv von dem Rank Stehelen der am meisten gearbeitet hat
    # Global von dem meist gearbeiteten Rank zu stehlen
    # Global zufallig
    # check correctness of output


for cfg in ["expb"]:
    datas = []

    jobtypes = jobtypes + ["bm"]
    for job in jobtypes:
        data = []

        prefix = job

        matching_dirs = filter_prefix(cfg + "-" + prefix, all_dirs)

        for dirname in matching_dirs:
            for nodecount in nodecounts:
                # print(dirname)
                outfiles = get_files(basepath + dirname)
                # print(outfiles)
                outfiles = filter_suffix(suffix, outfiles)
                outfiles = filter_prefix(cfg + "-" + nodecount + "-", outfiles)
                outfiles.sort()
                # print(outfiles)
                filename = outfiles[-1]
                found = False
                with open(basepath + dirname + "/" + filename, 'r') as output_file:
                    #print(basepath + dirname + "/" + filename)
                    for line in output_file:
                        if "Elapsed time (via clock_gettime):" in line:
                            tokens = line.split()
                            for token in tokens:
                                if check_float(token):
                                    if token != "9" and token != "18" and float(token) != 7.0:
                                        data.append(float(token))
                                        found = True
                    if not found:
                        data.append(0.0)
                        found = True

                if int(nodecount) > 1:
                    communication_stats(int(nodecount), basepath + dirname + "/" + filename, job)

        # datas.append(data)
        print(job + ": ", data)
        datas.append(data)

    #print(datas)

    with matplotlib.backends.backend_pdf.PdfPages("seissol_" + cfg + "_cpu_times.pdf") as pdf:
        plt.figure()
        plt.title("Time-to-solution:")

        for i in range(len(datas)):
            data = datas[i]
            label = jobtypes[i]
            plt.scatter(nodecounts, data, label=label)
            #plt.ylim(bottom=1)

        plt.legend()
        pdf.savefig()

    # Prediktiv von dem Rank Stehelen der am meisten gearbeitet hat
    # Global von dem meist gearbeiteten Rank zu stehlen
    # Global zufallig
    # check correctness of output



