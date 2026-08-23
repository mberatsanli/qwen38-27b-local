#!/bin/bash
# usage: gpumon.sh <outfile> ; samples every 200ms until killed
out=$1
echo "ts,mem_used_mib,util_gpu,util_mem,power_w,sm_clk,temp" > "$out"
while true; do
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu \
    --format=csv,noheader,nounits >> "$out"
  sleep 0.2
done
