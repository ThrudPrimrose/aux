import h5py
import numpy


def read_datasets(path):
    f1 = h5py.File(path, 'r')
    #print("Keys: %s" % f1.keys())
    group = list(f1.keys())[0]

    # print(group)
    print("Datasets: %s" % f1[group].keys())

    ds_data = f1[group]['clustering']  # returns HDF5 dataset object
    # print(ds_data)
    #print(ds_data.shape, ds_data.dtype)
    clustering = f1[group]['clustering'][:]  # adding [:] returns a numpy array
    # print(clustering)

    ds_data = f1[group]['connect']  # returns HDF5 dataset object
    # print(ds_data)
    #print(ds_data.shape, ds_data.dtype)
    connect = f1[group]['connect'][:]  # adding [:] returns a numpy array
    # print(connect)

    ds_data = f1[group]['partition']  # returns HDF5 dataset object
    # print(ds_data)
    #print(ds_data.shape, ds_data.dtype)
    partition = f1[group]['partition'][:]  # adding [:] returns a numpy array
    # print(partition)

    return (connect, clustering, partition)


def is_neighbor(v1, v2, v3, v4, w1, w2, w3, w4):
    has_v1 = int(v1 == w1 or v1 == w2 or v1 == w3 or v1 == w4)
    has_v2 = int(v2 == w1 or v2 == w2 or v2 == w3 or v2 == w4)
    has_v3 = int(v3 == w1 or v3 == w2 or v3 == w3 or v3 == w4)
    has_v4 = int(v4 == w1 or v4 == w2 or v4 == w3 or v4 == w4)

    sm = has_v1 + has_v2 + has_v3 + has_v4

    if sm == 3:
        return True
    else:
        # if sm == 4 -> same element, if sm < 3 -> not neighbors
        return False


def floor(a, b) -> int:
    if b >= a:
        return 0
    else:
        return a - b


def calc_good_and_bad_cuts(connect, clustering, partition):
    print('Begin')

    good_cut = 0
    bad_cut = 0

    start_early_index = 5
    size = connect.size//4

    # get vertices of
    for num1 in range(0, size):
        (v1, v2, v3, v4) = connect[num1]
        neighbors_found = 0
        beg = floor(num1, start_early_index)

        for num2 in range(beg, size):
            # print(connect.size, num1, num2)
            (w1, w2, w3, w4) = connect[num2]
            # print(w1, w2, w3, w4)
            if is_neighbor(v1, v2, v3, v4, w1, w2, w3, w4):
                neighbors_found += 1
                c1 = clustering[num1]
                c2 = clustering[num2]
                p1 = partition[num1]
                p2 = partition[num2]
                if p2 != p1:
                    if c1 == c2:
                        good_cut += 1
                    else:
                        bad_cut += 1

            if neighbors_found == 4:
                continue

        if floor(num1, start_early_index) > 0:
            for num2 in range(0, beg):
                (w1, w2, w3, w4) = connect[num2]
                if is_neighbor(v1, v2, v3, v4, w1, w2, w3, w4):
                    neighbors_found += 1
                    c1 = clustering[num1]
                    c2 = clustering[num2]
                    p1 = partition[num1]
                    p2 = partition[num2]
                    if p2 != p1:
                        if c1 == c2:
                            good_cut += 1
                        else:
                            bad_cut += 1

                if neighbors_found == 4:
                    continue

        if num1 % 200 == 0:
            #print('\x1b[1A' + '\x1b[2K\r')
            print(str(num1) + " / " + str(size), end='')

    return (good_cut, bad_cut)


def sort_zipped(a, b, c):
    connect, clustering, partition = zip(
        *sorted(zip(a, b, c), key=lambda v: v[0][0]))
    connect = numpy.array(connect)
    clustering = numpy.array(clustering)
    partition = numpy.array(partition)

    # print(connect)
    # print(clustering)
    # print(partition)

    return (connect, clustering, partition)


def calc_cuts_and_print_info(name):

    (connect, clustering, partition) = read_datasets(name)
    (connect, clustering, partition) = sort_zipped(
        connect, clustering, partition)

    (good, bad) = calc_good_and_bad_cuts(connect, clustering, partition)
    # f2 = h5py.File('output2/tpv5_cell.h5', 'r')
    print("From file: " + str(name) + ": " + str(good) +
          " of cuts were good and " + str(bad) + " bad")


def main():
    f1name = 'output1/tpv5_cell.h5'
    f2name = 'output2/tpv5_cell.h5'

    calc_cuts_and_print_info(f1name)
    calc_cuts_and_print_info(f2name)


if __name__ == "__main__":
    main()
