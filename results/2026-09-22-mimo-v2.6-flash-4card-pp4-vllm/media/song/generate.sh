#!/bin/bash
# Generate song takes with YuE2 (yue2_infer 0.1.5): 3 styles x 2 seeds, one GPU each, run in parallel.
# Chosen take: style mimo_synthpop, seed 11 (clearest words at a 35 s length).
# Usage: YUE2_DIR=/path/to/yue2 ./generate.sh
cd "${YUE2_DIR:?set YUE2_DIR to your yue2 install}"
for style in mimo_synthpop mimo_kpop mimo_chiptune; do
  (
    for seed in 11 124; do
      out=takes/${style}_s${seed}
      mkdir -p $out
      .venv/bin/yue2 generate --device cuda --cot full \
        --style "$(jq -r .style "$OLDPWD/$style.json")" --lyrics-file "$OLDPWD/lyrics.txt" \
        --seed $seed --output $out > $out/run.log 2>&1
    done
  ) &
done
wait
