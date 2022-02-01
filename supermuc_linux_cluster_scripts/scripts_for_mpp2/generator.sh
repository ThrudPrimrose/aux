#!/bin/bash

#predefined sizes

#WARNING 
# this script uses the environment variables set in set_vars_for_jobs.sh

#base time is the needed for 1000x1000-500-2 example to reach 1 sec (is ~150)
basetime=480 
#convert_time does not work for every size, check the generator and the function for the details

#convert from string to array
IFS=' ' read -r -a jobtypes <<< "$s_jobtypes"
echo ${jobtypes[*]}
IFS=' ' read -r -a sizes <<< "$s_sizes"
echo ${sizes[*]}
IFS=' ' read -r -a nodecounts <<< "$s_nodecounts"
echo ${nodecounts[*]}
IFS=' ' read -r -a endtimes <<< "$s_endtimes"
echo ${endtimes[*]}
IFS=' ' read -r -a patches <<< "$s_patches"
echo ${patches[*]}

rm -r -f ${jobscriptdir}
#create ${jobscriptdir} 
if [ -d ./${jobscriptdir}/ ]
then
    echo "${jobscriptdir} exists"
else
    echo "create ${jobscriptdir}"
    mkdir ${jobscriptdir}
    chmod -R 775 ${jobscriptdir}
fi

#check for scratch
if [ -z "$SCRATCH" ]
then
    echo "scratch not set"
    SCRATCH=${PWD}
fi

#create ${workdir} in scratch
if [ -d ${SCRATCH}/${workdir} ]
then
    echo "${workdir} in scracth exists"
else
    mkdir ${SCRATCH}/${workdir}
    chmod -R 775 ${SCRATCH}/${workdir}
fi

cd ${jobscriptdir}

#convert_time does not work for every size, check the generator and the function for the details
function convert_time {
    multiplier=$(( size / 2000 ))
    multiplier=$(( multiplier * multiplier ))
    limit=$(( limit * multiplier ))

    limit=$(( limit / nodecount ))

    limit=$(( limit * endtime ))

    hours=$(( limit / 3600 ))
    mins=$(( limit % 3600 ))
    secs=$(( mins % 60 ))
    mins=$(( mins / 60 ))

    if [[ "$hours" -ge "0" && "$hours" -le "9" ]]
    then
        hours=0${hours}
    fi
    if [[ "$mins" -ge "0" && "$mins" -le "9" ]]
    then
        mins=0${mins}
    fi
    if [[ "$secs" -ge "0" && "$secs" -le "9" ]]
    then
        secs=0${secs}
    fi

    if [[ "$hours" -ge 8 ]]
    then
        hours=07
        mins=59
        secs=59
    fi
}

#generate folders for each task
for size in ${sizes[@]}
do
    for nodecount in ${nodecounts[@]}
    do
        for endtime in ${endtimes[@]}
        do
            for job in ${jobtypes[@]}
            do
                for patchsize in ${patches[@]}
                do
                    
                    str=pond-${job}-${nodecount}-${patchsize}-${size}x${size}-${endtime}
                    if [ -d ${str} ]
                    then
                        #rm -r ${str}
                        echo "old results for "${str}" exists, did not touch"
                    else
                        mkdir ${str}
                    fi

                    cd ${str}

                    scriptname=pond-${job}.sh
                    
                    #write the modified base script
                    if [ -f "$scriptname" ]
                    then
                        rm $scriptname
                    fi

                    corecountforjob=$(( corecountpernode * nodecount ))

                    limit=basetime
                    convert_time

                    qos=""
                    partition=""
                    docomment=""
                    clusters=""

                    if [[ "$nodecount" -le 2 ]]
                    then
                        qos=""
                        partition="cm2_tiny"
                        clusters="cm2_tiny"
                    else
                        if [[ "$nodecount" -le 24  ]]
                        then
                            qos="cm2_std"
                            partition="cm2_std"
                            clusters="cm2"
                            docomment="SBATCH"
                        else
                            qos="cm2_large"
                            partition="cm2_large"
                            clusters="cm2"
                            docomment="SBATCH"
                        fi
                    fi

cat <<EOF >$scriptname
#!/bin/bash
#SBATCH -J ${str}
#SBATCH -o ${SCRATCH}/${workdir}/${str}/%x.%j.out
#SBATCH -e ${SCRATCH}/${workdir}/${str}/%x.%j.err
#SBATCH -D ./
#Notification and type
#SBATCH --mail-type=end,fail,timeout
#SBATCH --mail-user=yakup.paradox@gmail.com
# Wall clock limit:
#SBATCH --time=${hours}:${mins}:${secs}
#SBATCH --no-requeue
#Setup of execution environment
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --clusters=${clusters}
#${docomment} --qos=${qos}
#SBATCH --partition=${partition}
#SBATCH --nodes=${nodecount}
#SBATCH --ntasks-per-node=${corecountpernode}

module load slurm_setup
module load netcdf-hdf5-all
module load metis

export UPCXX_INSTALL=~/upcxx-intel-mpp2
export PATH=$PATH:~/upcxx-intel-mpp2/bin
export GASNET_PHYSMEM_MAX='40 GB'
export GASNET_BACKTRACE=1
export REPARTITIONING_INTERVAL=60.0

#41 GB is the max number in mpp2 (both _inter and _tiny)

export numproc=$(( corecountpernode * nodecount ))

upcxx-run -N ${nodecount} -n $(( corecountpernode * nodecount )) -shared-heap 1GB /dss/dsshome1/lxc05/ge69xij2/ActorHPC/./pond-${job} -x ${size} -y ${size}  -p ${patchsize} -c 10 --scenario 3 -o ${SCRATCH}/${workdir}/${str}/out/out -e ${endtime}

cat > ${SCRATCH}/${workdir}/${str}/finished.txt

EOF
                    chmod 775 $scriptname
                    cd ..
                done
            done
        done
    done
done

cd ..
echo ${pwd}
echo "done generating script folders"

echo "generate folders in scratch"
for size in ${sizes[@]}
do
    for nodecount in ${nodecounts[@]}
    do
        for endtime in ${endtimes[@]}
        do
            for job in ${jobtypes[@]}
            do
                for patchsize in ${patches[@]}
                do
                    str=pond-${job}-${nodecount}-${patchsize}-${size}x${size}-${endtime}
                    if [ -d ${SCRATCH}/${workdir}/${str} ]
                    then
                        #rm -r ${str}
                        echo "old results for "${SCRATCH}/${workdir}/${str}" exists, did not touch"
                    else
                        mkdir ${SCRATCH}/${workdir}/${str}
                        mkdir ${SCRATCH}/${workdir}/${str}/out
                    fi
                    chmod -R 775 ${SCRATCH}/${workdir}/${str}
                done
            done
        done
    done
done
