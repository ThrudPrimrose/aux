import datetime
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib import colors as mcolors
import re
import numpy as np

datas = []
message_sent_counts = []
message_sent_sizes = []
variance_message_sent_counts = []
variance_message_sent_sizes = []

base = "cpuruns/results-loh_600-"
suffix = ".out"

"""
consts = ["0.85", "1.00", "1.15", "1.30", "1.45", "1.60", "1.85", "2.00", "2.15", "2.30", "2.45",
          "2.60", "2.85", "3.00", "3.15", "3.30", "3.45", "3.60", "3.85", "4.00"]
"""

consts = ["1.00", "2.00", "3.00"]
job_types = ["NAIVE", "160", "COMM APPROX"]


today = datetime.datetime.now()
day_string = today.strftime("%d")
day_f_string = str(day_string)
#print(day_f_string)


"""
        ss << "Rank " << rank << " statistics of Isend and Irecv calls\n";
        ss << "Rank " << rank << " has sent: " << messages_sent.load() << " messages\n";
        ss << "Rank " << rank << " has sent: " << total_bytes_sent.load() << " bytes\n";
        ss << "Rank " << rank << " sent message size is on average: " << static_cast<double>(total_bytes_sent.load()) 
        / static_cast<double>(messages_sent.load()) << "\n";
        ss << "Rank " << rank << " has received: " << messages_received.load() << " messages\n";
        ss << "Rank " << rank << " has received: " << total_bytes_received.load() << " bytes\n";
        ss << "Rank " << rank << " received message size is on average: " << static_cast<double>(total_bytes_received.load()) 
        / static_cast<double>(messages_received.load()) << "\n";
        ss << "Rank " << rank << " called test: " << test_call.load();
"""

#change this depending on the ranks
rankcount = 4


def find_jobtype_begin(name, jobtype):
    with open(name, 'r') as output_file:
        for num, line in enumerate(output_file, 1):
            if jobtype in line:
                # print(jobtype, " starts at: ", num)
                return num
    return -1


def find_jobtype_end(beg_offset, name, jobtype):
    with open(name, 'r') as output_file:
        for num, line in enumerate(output_file, 1):
            if "========================================================================" in line and num > beg_offset:
                # print(jobtype, " ends at: ", num)
                return num
    return -1


def communication_stats(beg_offset, end_offset, name, jobtype):
    sentmsg = [0] * rankcount
    sentsize = [0.0] * rankcount
    ranklist = range(rankcount)

    with open(name, 'r') as fp:
        for i, line in enumerate(fp):
            if i >= beg_offset and i <= end_offset:
                for r in ranklist:
                    if "Rank " + str(r) + " has sent: " in line and "messages" in line:
                        # find all integers, first one is rank number so take the second one
                        sentmsg[r] = [int(s)
                                      for s in line.split() if s.isdigit()][1]
                    if "Rank " + str(r) + " has sent: " in line and "bytes" in line:
                        # there should be only one floating point number in the lane so we will take it
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

    return (msg_mean, byte_mean, msg_var, byte_var)


def check_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


for c in consts:
    name = base + c + suffix
    mm = list() 
    ms = list()
    vm = list()
    vs = list()

    for jt in job_types:

        
        beg = find_jobtype_begin(name, jt)
        end = find_jobtype_end(beg, name, jt)

        (a,b,c,d) = communication_stats(beg, end, name, jt)
        mm.append(a)
        ms.append(b)
        vm.append(c)
        vs.append(d)

    message_sent_counts.append(mm)
    message_sent_sizes.append(ms)
    variance_message_sent_counts.append(vm)
    variance_message_sent_sizes.append(vs)


for c in consts:
    name = base + c + suffix
    f_vec = []

    with open(name, 'r') as output_file:
        for line in output_file:
            if "Elapsed time (via clock_gettime):" in line:
                tokens = line.split()
                for token in tokens:
                    if check_float(token):
                        if token != day_f_string:
                            f_vec.append(float(token))

    datas.append(f_vec)

print(datas)


def plot_bars(name, dataset, job_types, configurations):
    with matplotlib.backends.backend_pdf.PdfPages(name) as pdf:
        for i in range(len(dataset)):
            title = configurations[i]
            y_cors = dataset[i]
            plt.figure()
            plt.title(title)
            print(y_cors)
            if (len(y_cors) == len(job_types)):
                colours = ['steelblue'] * len(y_cors)
                max_ind = y_cors.index(max(y_cors))
                min_ind = y_cors.index(min(y_cors))
                colours[max_ind] = 'orangered'
                colours[min_ind] = 'mediumturquoise'
                plt.bar(job_types, y_cors, color=colours)
            pdf.savefig()


def plot_comm(name, dataset, job_types, configurations):
    with matplotlib.backends.backend_pdf.PdfPages(name) as pdf:
        for i in range(len(dataset)):
            title = configurations[i]
            y_cors1 = message_sent_counts[i]
            y_cors2 = message_sent_sizes[i]
            y_cors3 = variance_message_sent_counts[i]
            y_cors4 = variance_message_sent_sizes[i]
            
            y_cors_all = [y_cors1,y_cors2,y_cors3,y_cors4]
            aux_titles = ["Average msg sent count", "Average msg size", "Variance msg sent count", "Variance msg size"]

            for (y_cors, aux_tit) in zip(y_cors_all, aux_titles):  
                tt = title + " " + aux_tit
                plt.figure()
                plt.title(tt)
                print(y_cors)
                if (len(y_cors) == len(job_types)):
                    colours = ['steelblue'] * len(y_cors)
                    max_ind = y_cors.index(max(y_cors))
                    min_ind = y_cors.index(min(y_cors))
                    colours[max_ind] = 'orangered'
                    colours[min_ind] = 'mediumturquoise'
                    plt.bar(job_types, y_cors, color=colours)
                pdf.savefig()

plot_bars("SeisSol_runtime_loh_600.pdf", datas, job_types, consts)
plot_comm("SeisSol_communication_loh_600.pdf", datas, job_types, consts)
