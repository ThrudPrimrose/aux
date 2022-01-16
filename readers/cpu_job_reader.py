from os import listdir
from os.path import isfile, join

datas = []

consts = ["0.80", "0.85", "1.00", "1.15", "1.30", "1.45", "1.60", "1.85", "2.00", "2.15", "2.30", "2.45",
          "2.60", "2.85", "3.00", "3.15", "3.30", "3.45", "3.60", "3.85", "4.00"]

jobtypes = ["eb", "exp"]
nodecounts = ["1", "2", "4", "6", "8"]


def get_files(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles


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


basepath = "seissol_cpu_runs/out/"
all_files = get_files(basepath)

for job in jobtypes:
    data = []

    for nodecount in nodecounts:
        prefix = job + "-" + nodecount
        # print(prefix)
        suffix = ".out"
        matching_files = filter_suffix(
            suffix, filter_prefix(prefix, all_files))
        # print(matching_files)

        for name in matching_files:
            print(name)
            with open(basepath + name, 'r') as output_file:
                for line in output_file:
                    if "Elapsed time (via clock_gettime):" in line:
                        tokens = line.split()
                        for token in tokens:
                            if check_float(token) and token != "13.0":
                                if float(token) != 13.0:
                                    data.append(float(token))

    datas.append(data)
    print(data)
