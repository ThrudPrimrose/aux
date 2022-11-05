import datetime
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib import colors as mcolors

datas = []

base = "run3/results-"
suffix = ".out"

consts = ["0.80", "0.85", "1.00", "1.15", "1.30", "1.45", "1.60", "1.85", "2.00", "2.15", "2.30", "2.45",
          "2.60", "2.85", "3.00", "3.15", "3.30", "3.45", "3.60", "3.85", "4.00"]

today = datetime.datetime.now()
day_string = today.strftime("%d")
day_f_string = str(day_string)
print(day_f_string)


def check_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


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


job_types = ["Naive", "140", "160", "180", "200", "CommApprox"]


def plot_bars(name, dataset, job_types, configurations):
    with matplotlib.backends.backend_pdf.PdfPages(name) as pdf:
        for i in range(len(dataset)):
            title = configurations[i]
            y_cors = dataset[i]
            plt.figure()
            plt.title(title)
            # print(y_cors)
            if (len(y_cors) == len(job_types)):
                colours = ['steelblue'] * len(y_cors)
                max_ind = y_cors.index(max(y_cors))
                min_ind = y_cors.index(min(y_cors))
                colours[max_ind] = 'orangered'
                colours[min_ind] = 'mediumturquoise'
                plt.bar(job_types, y_cors, color=colours)
            pdf.savefig()


plot_bars("SeisSol_runtime.pdf", datas, job_types, consts)
