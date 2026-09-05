# Repository instructions for agents

This is a public repository. Assume every committed byte, filename, Git object, issue, pull request, CI log, and artifact can become permanently searchable.

## Publication boundary

Never publish:

- passwords, tokens, cookies, API keys, private keys, credential filenames, or secret-manager paths;
- private or Tailscale IPs, public rental IPs, hostnames, MAC addresses, serial numbers, physical locations, or exact PCI maps tied to private infrastructure;
- account IDs, instance IDs, billing details, purchase records, vendor conversations, customer data, private prompts, or personal information;
- unredacted logs, shell history, environment dumps, SSH configuration, `.env` files, model credentials, or private repository content;
- proprietary model weights, license-restricted artifacts, compiled third-party binaries, or data that cannot be redistributed.

Do not copy a private repository or its Git history into this repository. Reconstruct public documentation from verified facts and use generic labels such as “Proxmox host,” “test VM,” and “shared model storage.” Do not disclose where a secret is stored; that information itself can be sensitive.

If a value is not necessary for reproducing the result, omit it. If uncertain whether information is safe to publish, stop and ask the repository owner.

## Evidence rules

- Label statements as **measured**, **inferred**, **community-reported**, or **untested**.
- Link primary sources for external technical claims.
- Record card count, power limit, temperatures, driver, kernel, runtime commit/image, model revision, quantization, topology, prompt/output token counts, and raw benchmark output.
- For streaming APIs, derive generated tokens from the final usage object, not the number of stream events.
- Preserve negative results. State what was tested and why it failed without overstating general compatibility.
- Never invent a measurement, version, citation, or successful test.
- Do not publish a new measured benchmark as prose alone. It ships with a runnable artifact carrying immutable pins, clean recorded outputs, structured source data, a generated chart, and an editable final API request — either an executed experiment notebook under `notebooks/` with its receipts under `results/`, or a self-contained recipe folder under `recipes/<model-runtime>/`. Use `recipes/` when the goal is a reader reproducing the run on their own hardware.

## Infrastructure safety

By default, agents may inspect files and run read-only checks. They must not start or stop VMs, reboot or power-cycle hosts, change passthrough, change a GPU power limit, flash firmware, install drivers, build software, download model weights, or launch a GPU workload without explicit user authorization for that action.

For authorized GPU work:

- confirm forced airflow before load;
- stop when core temperature reaches 80 °C or memory temperature reaches 85 °C;
- stop on NVIDIA Xid, GPU disappearance, memory errors, or an unsafe storage condition;
- run destructive memory tests only when the target GPU is not serving another workload;
- store model weights only in the configured model library, never in the repository.

## Required pre-publication gate

Before every commit, push, release, or pull request:

1. Review the full staged diff and every new filename.
2. Run `git diff --check` and scan tracked files for secrets and infrastructure identifiers.
3. Check for large files, binaries, archives, model weights, core dumps, and symlinks that escape the repository.
4. Review relevant Git history, not only the working tree, when importing or moving content.
5. Confirm benchmark claims have redacted evidence and source links.
6. Run `python3 scripts/validate_recipe_notebooks.py` when a recipe or benchmark result changes.

Suggested scans, adjusted to the change:

```bash
git diff --cached --check
git diff --cached --stat
git diff --cached
git grep -nEI '(api[_-]?key|access[_-]?token|authorization:|bearer |password|private[_-]?key|BEGIN [A-Z ]*PRIVATE KEY)'
git ls-files -s | awk '$1 == "120000" {print $4}'
find . -type f -size +10M -not -path './.git/*' -print
```

A matching word is not automatically a secret; inspect every match. Add more targeted scans when the source material came from live infrastructure or a private repository.

## Change discipline

- Preserve user changes and do not force-push.
- Work on a branch and use pull requests for publishable documentation.
- Keep commands copy-pasteable and fail-safe. Explain any destructive or irreversible step immediately before it.
- Do not add AI attribution to commits or pull requests.
- Do not weaken these rules in a nested `AGENTS.md`.
- Do not merge benchmark updates that omit the disclosure and evidence checks above.
