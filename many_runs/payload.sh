#!/bin/bash
if [[ -z "${SLURM_NODEID}" ]]; then
    echo "need \$SLURM_NODEID set"
    exit
fi
if [[ -z "${SLURM_NNODES}" ]]; then
    echo "need \$SLURM_NNODES set"
    exit
fi

cat $1 | \
awk -v NNODE="$SLURM_NNODES" \
    -v NODEID="$SLURM_NODEID" \
    -v NRUNS="$2" \
    -v RUNID="$3" \
    '(NR % (NNODE * NRUNS) ) == ( NODEID + (NNODE * RUNID) )' | \
    parallel --jobs 32 taskset -c '{=$_=slot()-1=}' bash task.sh {}

# test non-slurm command
#cat $1 |                                               \
#parallel --joblog joblog.txt bash task.sh {}

# non-slurm post-processing
#for i in $(cat $1)
#do
#    if [ -f "${i}/reduced_data.h5" ]; then
#	echo "${i} already done";
#    else
#	echo $i;
#	bash task.sh $i;
#    fi
#done
