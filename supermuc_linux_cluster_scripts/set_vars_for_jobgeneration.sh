#!/bin/bash

# dont forget to run this with 
# source set_vars_for_jobgeneration.sh

#need to serialize are array into strings and then get them back

#job types generate subfolders for everype
#_jobtypes=(static invasion staticlazy bulklazy steal lazysteal invasionsteal bulk)
_jobtypes=(steal lazy bulk bulklazy static steallazy hybrid hybridlazy)
_jobtypes=$( IFS=' '; printf '%s' "${_jobtypes[*]}" )
export s_jobtypes=$_jobtypes
#sizes of the total gitter SxS
_sizes=(12000)
_sizes=$( IFS=' '; printf '%s' "${_sizes[*]}" )
export s_sizes=$_sizes
#number of cpus per node
export corecountpernode=28
#how many nodes the job will run
_nodecounts=(1 2 4 6)
_nodecounts=$( IFS=' '; printf '%s' "${_nodecounts[*]}" )
export s_nodecounts=$_nodecounts
#end times for the simulation
_endtimes=(1)
_endtimes=$( IFS=' '; printf '%s' "${_endtimes[*]}" )
export s_endtimes=${_endtimes}
#an actor will have patchsize SxS
_patches=(250)
_patches=$( IFS=' '; printf '%s' "${_patches[*]}" )
export s_patches=${_patches}
#the name of folder to save the work
export workdir=strongscaling
#the upcxx_install path for cmake, if you have insstalled somewhere else then change this
export UPCXX_INSTALL=~/upcxx-intel-mpp2
#add upcxx path to the path, if you have installed somewhere else then change it
export PATH=$PATH:~/upcxx-intel-mpp2/bin
export jobscriptdir="jobscripts-ex"


