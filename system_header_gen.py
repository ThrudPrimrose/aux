import os
import shutil

rootdir = "src"

system_header_entries = set()


def remove_headers(path, is_cpp_file, include_precompiled_header):
    with open(path + ".temporary", "w") as tfile:        
        system_header_count = 0
        with open(path, "r") as file:    
            for line in file:
                if line.startswith("#include <"):
                    system_header_count += 1
            
        if include_precompiled_header or system_header_count > 0:
            tfile.write("#include \"StdAfx.hpp\"\n")

        with open(path, "r") as file:    
            ifdef_block = list()
            else_count  = 0
            i = 0
            for line in file:
                if line.startswith("#ifdef") or line.startswith("#if defined") \
                    or line.startswith("#ifndef") or line.startswith("#if !defined") \
                    or (line.startswith("#if") and ("==" in line or "<" in line or ">" in line )) \
                    or (len(ifdef_block) > 0 and line.startswith("#else")):
                    ifdef_block.append(line)
                    if (len(ifdef_block) > 0 and line.startswith("#else")):
                        else_count += 1
                if line.startswith("#endif"):
                    try:
                        if (ifdef_block[len(ifdef_block) - 1].startswith("#else")):
                            ifdef_block.pop()
                        ifdef_block.pop()
                    except Exception as e:
                        print("".join(ifdef_block))
                        print(f"Empty list problem in file: {path}, line: {i}, check it")
                        # exit()
                if line.startswith("#include <"):
                    s = "".join(ifdef_block)
                    s += line
                    s += "#endif\n" * (len(ifdef_block) - else_count)
                    print(f"In file: {path}: Found an include block:\n{s}")
                    system_header_entries.add(s)
                else:
                    tfile.write(line)
                i += 1
        
            #if system_header_count > 0:
            shutil.move(path + ".temporary", path)
            return system_header_count


for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        path = os.path.join(subdir, file)
        if path.endswith(".h") and  not "src/tarch/compiler" in path:
            h_path = path
            cpp_path = path.replace(".h", ".cpp")
            cpph_path = path.replace(".h", ".cpph")

            if os.path.exists(cpph_path):
                remove_headers(cpph_path, False, False)
            

            if os.path.exists(cpp_path):
                #Then remove systme headers (from h and cpp files both), include system headers in StdAfx.cpp
                #and include StdAfx.hpp in .cpp file
                #Remove the system headers, include StdAfx.hpp in h file
                system_headers_found = remove_headers(h_path, False, False)
                remove_headers(cpp_path, True, system_headers_found > 0)
                
            else:
                remove_headers(h_path, False, False)

with open("src/StdAfx.hpp", "w") as system_header_file:
    for line in system_header_entries:
        system_header_file.write(line) 