#!/bin/bash

BASEDIR=$(dirname "$0")
echo "Running at: $BASEDIR"

echo "Setting environment variables"
source ${BASEDIR}/../set_vars_for_jobgeneration.sh
echo ${jobtypes}

echo "Generating SLURM job scripts"
${BASEDIR}/generator.sh

echo "Compiling and copying executables"
${BASEDIR}/builder.sh

echo "Moving pond executables"
${BASEDIR}/move.sh

echo "Submitting jobs"
${BASEDIR}/../submitter.sh
