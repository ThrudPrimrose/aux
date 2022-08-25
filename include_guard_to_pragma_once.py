import sys
import tempfile
import shutil


def replace():
    path = sys.argv[1]

    found = False
    first_ifndef_index = -1
    first_define_index = -1
    last_endif_index = -1

    input_file = open(path, 'r')

    consecutive_empty_lines_at_end = 0

    i = 0
    for line in input_file:
        if "#ifndef" in line:
            first_ifndef_index = i
        if "#define" in line and first_ifndef_index == i-1:
            first_define_index = i
        if "#endif" in line and first_ifndef_index < i-1:
            last_endif_index = i
        i += 1

    input_file.seek(0)
    tmp = tempfile.NamedTemporaryFile(delete=False)
    i = 0
    with open(tmp.name, 'w') as output_file:
        for line in input_file:
            if i == first_ifndef_index:
                output_file.write("#pragma once\n")
                print('#pragma once')
            elif i != first_define_index and i != last_endif_index:
                if not (i == last_endif_index - 1 and (len(line) == 0 or line == "\n")):
                    output_file.write(line)

                print(line)
            i += 1

    print(tmp)
    print(tmp.name)
    # print(tmp.read())
    shutil.move(tmp.name, path)


def main():
    replace()


if __name__ == "__main__":
    main()
