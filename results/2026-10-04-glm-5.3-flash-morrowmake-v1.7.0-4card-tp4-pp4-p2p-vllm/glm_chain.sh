#!/bin/bash
# GLM v1.7.0 option chain. Watchdog: any card >= 80 C -> cap 140 W, stop server, abort.
R=<recipe>; B=<bench>; L=$B/glm_chain.log
( while sleep 5; do t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sort -n | tail -1); if [ "$t" -ge 80 ]; then echo "$(date -u +%T) WATCHDOG temp $t C: abort" >> $L; nvidia-smi -pl 140 >/dev/null; docker stop -t 30 glm53-flash >/dev/null; kill -TERM $$; exit; fi; done ) & WD=$!
for cfg in "glm-v170-tp4-nosys-140:140:LAYOUT=tp4 GLM5_NCCL_P2P_SYS=0" "glm-v170-pp4-140:140:LAYOUT=pp4" "glm-v170-tp4-nosys-180:180:LAYOUT=tp4 GLM5_NCCL_P2P_SYS=0" "glm-v170-pp4-180:180:LAYOUT=pp4"; do
  name=${cfg%%:*}; rest=${cfg#*:}; pw=${rest%%:*}; envs=${rest#*:}
  nvidia-smi -pl $pw >/dev/null; echo "$(date -u +%T) START $name (pl=$pw W, $envs)" >> $L
  (cd $R && env $envs ./start.sh restart > $B/$name-start.log 2>&1)
  until curl -sf -m 5 localhost:8030/health >/dev/null; do docker ps -q -f name=glm53-flash | grep -q . || { echo "$(date -u +%T) DIED $name" >> $L; continue 2; }; sleep 15; done
  echo "$(date -u +%T) HEALTHY $name $(grep "\[p2p\]" $R/logs/serve.log | tail -1 | cut -c1-60)" >> $L
  cp $R/logs/serve.log $B/$name-serve.log 2>/dev/null
  $B/bench_glm.sh $name; echo "$(date -u +%T) END $name" >> $L
done
nvidia-smi -pl 140 >/dev/null; echo "$(date -u +%T) CHAIN-DONE (power cap back to 140 W)" >> $L
kill $WD
