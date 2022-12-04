import os

node_counts = [1, 2, 4, 8, 16, 32]
core_count_per_node = 28
case_name = "StrongScaling"

# Update this variable depending on the run time of a small job
# e.g. 2D Cavity without output for
# 1 Node for 100x100 w. final time 10 takes 35 seconds
core_base_time = 50

# For strong scaling let's run a job of end time 20,
# and 1000x1000 on 10, 5x5 for each dimension 2 for doubling end time
base_time = core_base_time*10*10*2

def convert_seconds_to_string(seconds):
    hours = str(seconds // 3600)
    minutes = str((seconds % 3600) // 60)
    seconds = str(seconds % 60)
    if len(hours) >= 3:
        hours = "99"
    elif len(hours) == 1:
        hours = "0" + hours
    if len(minutes) == 1:
        minutes = "0" + minutes
    if len(seconds) == 1:
        seconds = "0" + seconds

    return f"{hours}:{minutes}:{seconds}"

for node_count in node_counts:
    job_xml_name = f"Cavity2D-{node_count}node"
    xml_string = f"""<?xml version="1.0" encoding="utf-8"?>
<configuration>
    <flow Re="500" />
    <simulation finalTime="20" >
        <type>dns</type>
        <scenario>cavity</scenario>
    </simulation>
    <timestep dt="1" tau="0.5" />
    <solver gamma="0.5" />
    <geometry dim="2"
      lengthX="1.0" lengthY="1.0" lengthZ="1.0"
      sizeX="1000" sizeY="1000" sizeZ="20"
    >
      <mesh>uniform</mesh>
    </geometry>
    <environment gx="0" gy="0" gz="0" />
    <walls>
        <left>
            <vector x="0" y="0" z="0" />
        </left>
        <right>
            <vector x="0" y="0" z="0" />
        </right>
        <top>
            <vector x="1" y="0" z="0" />
        </top>
        <bottom>
            <vector x="0" y="0" z="0" />
        </bottom>
        <front>
            <vector x="0" y="0" z="0" />
        </front>
        <back>
            <vector x="0" y="0" z="0" />
        </back>
    </walls>
    <vtk interval="2147483647">Cavity2D</vtk>
    <stdOut interval="2147483647" />
    <parallel numProcessorsX="{node_count}" numProcessorsY="{core_count_per_node}" numProcessorsZ="1" />
</configuration>
"""

    #path = case_name + "-" + str(node_count)
    #if not os.path.exists(path):
    #    os.makedirs(path)
    #xmlfile = open(path + "/" + job_xml_name + ".xml", "w")
    xmlfile = open(job_xml_name + ".xml", "w")
    xmlfile.write(xml_string)
    xmlfile.close()

    job_name = f"strong-scaling-{node_count}node"

    if node_count <= 2:
        cluster = "cm2_tiny"
        have_qos = ""
        qos = ""
        partition = "cm2_tiny"
    elif node_count <= 24:
        cluster = "cm2"
        have_qos = "SBATCH"
        qos = "cm2_std"
        partition = "cm2_std"
    else:
        cluster = "cm2"
        have_qos = "SBATCH"
        qos = "cm2_large"
        partition = "cm2_large"

    time_limit = convert_seconds_to_string(int(base_time / node_count))

    job_script = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH -D ./
#SBATCH --mail-type=end,fail,timeout
#SBATCH --mail-user=yakup.paradox@gmail.com
#SBATCH --time={time_limit}
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --clusters={cluster}
#{have_qos} --qos={qos}
#SBATCH --partition={partition}
#SBATCH --nodes={node_count}
#SBATCH --ntasks-per-node={core_count_per_node}

module load slurm_setup
module unload intel-mpi/2019-intel
module unload intel-oneapi-compilers/2021.4.0
module unload intel-mkl/2020
module load cmake
module load gcc/11
module load petsc/3.17.2-gcc11-ompi-real
module load openmpi/4.1.2-gcc11

mpirun -n {node_count * core_count_per_node} ${{HOME}}/ns-eof/build/NS-EOF-Runner \\
    ${{HOME}}/runs/{job_xml_name}.xml
"""

    #jobfile = open(path + "/" + job_name + ".job", "w")
    jobfile = open(job_name + ".job", "w")
    jobfile.write(job_script)
    jobfile.close()

    #os.system("sbatch " + path + "/" + job_name + ".job")
    os.system("sbatch " + job_name + ".job")