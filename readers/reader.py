data = []

base = "results-"
suffix = ".out"

consts = ["0.80", "0.85", "1.00", "1.15", "1.30", "1.45", "1.60", "1.85", "2.00", "2.15", "2.30", "2.45",
          "2.60", "2.85", "3.00", "3.15", "3.30", "3.45", "3.60", "3.85", "4.00"]


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
                    if check_float(token) and token != "12.0":
                        if float(token) != 12.0:
                            f_vec.append(float(token))

    data.append(f_vec)

print(data)
