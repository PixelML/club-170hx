| model                          |       size |     params | backend    | ngl |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------------- | --------------: | -------------------: |
| qwen4exp A3B Q3_K - Medium     |  83.80 GiB |   176.94 B | CUDA       |  99 |   1 | per_layer_token_embd=CPU |           pp512 |         76.27 ± 8.09 |
| qwen4exp A3B Q3_K - Medium     |  83.80 GiB |   176.94 B | CUDA       |  99 |   1 | per_layer_token_embd=CPU |           tg128 |         34.79 ± 0.64 |
