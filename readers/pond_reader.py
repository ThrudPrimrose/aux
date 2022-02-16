import os
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

datas = []

jobtypes = ["static", "lazy", "bulk", "bulklazy",
            "steal", "steallazy", "hybrid", "hybridlazy"]
nodecounts = ["1", "2", "4", "6"]


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


basepath = "strongscaling/"
suffix = ".out"
all_dirs = get_dirs(basepath)

for job in jobtypes:
    data = []

    for nodecount in nodecounts:
        prefix = "pond-" + job + "-" + nodecount
        # print(prefix)

        matching_dirs = filter_prefix(prefix, all_dirs)
        # print(matching_dirs)

        for dirname in matching_dirs:
            # print(name)
            outfiles = get_files(basepath + dirname)
            outfiles = filter_suffix(suffix, outfiles)
            outfiles.sort()
            # print(outfiles)
            filename = outfiles[-1]
            found = False
            with open(basepath + dirname + "/" + filename, 'r') as output_file:
                for line in output_file:
                    if "time-to-solution:" in line and not found:
                        tokens = line.split(":")
                        for token in tokens:
                            if check_float(token):
                                data.append(float(token))
                                found = True
                if not found:
                    data.append(0.0)
                    found = True

    # datas.append(data)
    print(job + ": ", data)
    datas.append(data)

print(datas)


with matplotlib.backends.backend_pdf.PdfPages("pond_times.pdf") as pdf:
    plt.figure()
    plt.title("Time-to-solution:")

    for i in range(len(datas)):
        data = datas[i]
        label = jobtypes[i]
        plt.plot(nodecounts, data, label=label)

    plt.legend()
    pdf.savefig()

    # Prediktiv von dem Rank Stehelen der am meisten gearbeitet hat
    # Global von dem meist gearbeiteten Rank zu stehlen
    # Global zufallig
    # check correctness of output
