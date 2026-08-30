# Contributing

Contributions are welcome: reproducible benchmarks, card-validation data, cooling experiments, compatibility results, and corrections are all useful.

## Before opening a pull request

1. Read [AGENTS.md](AGENTS.md) and remove private infrastructure details and secrets.
2. Use the format in [results/README.md](results/README.md) for performance claims.
3. Separate measured results from assumptions and community reports.
4. Run shell checks where applicable: `bash -n scripts/*.sh scripts/qc/*.sh`.
5. Confirm links, commands, and the complete diff.

Small, evidence-backed changes are easier to review than broad claims. Negative results are welcome when the environment and failure mode are documented.

## Commit style

Use a short Conventional Commit subject, for example:

```text
docs: add three-card pipeline benchmark
fix: map PCIe checks to the correct GPU
```

## Safety

Never run a memory-destructive test on a GPU that is serving another process. Never benchmark a passive card without directed airflow and live temperature monitoring.
