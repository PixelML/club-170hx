#!/bin/bash
# sample GPU power / clocks / temperatures once per second into $1 until killed
nvidia-smi --query-gpu=timestamp,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,temperature.memory,utilization.gpu,memory.used --format=csv -l 1 > "$1"
