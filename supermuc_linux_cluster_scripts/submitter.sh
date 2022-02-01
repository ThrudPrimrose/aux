#!/bin/bash

IFS=' ' read -r -a jobtypes <<< "$s_jobtypes"
#echo ${jobtypes[*]}

#supermuc has a job limit, it was 50 for tiny jobs in mpp3
maxjobs=50

#counter going from 0 to maxjobs
counter=0

#output directory in ${SCRATCH}

cd ${jobscriptdir}

for t in ${jobtypes[@]}
do
    for i in pond-${t}-*/
    do
        #check if the script did ${workdir}et
            if [ ${counter} -le "50" ]
            then
                cd ${i}
                sbatch pond-${t}.sh
                counter=$(( counter + 1 ))
                cd ..
            else
                echo "max number of submissions reached"
                exit 
            fi           
    done
done

cd ..
echo "submitted ${counter} jobs"



