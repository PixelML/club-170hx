# Cooling and power

CMP 170HX cards are passive server accelerators. An open case and a room fan are not automatically enough: air must be forced through the heatsink fins, and exhaust must not recirculate into the intake.

## Idle heat is real heat

On 2026-08-30, after the build described in the [hardware guide](HARDWARE.md#our-four-card-rig) was finished, the four-card rig idled with a 180 W limit per card and the blower running:

| Metric | Measured idle snapshot (n=1) |
|---|---|
| Core temperature | 37–38 °C |
| Memory temperature | 41–51 °C |
| Board power per card | 33.71–37.69 W |
| Group power, four cards | about 141 W |
| GPU utilization / VRAM used | 0% / 0 MiB on every card |

This is one timestamped snapshot; check live values with the `nvidia-smi` query command under [practical policies](#practical-policies). Ambient temperature and blower RPM were not recorded, so treat these ranges as observations from this frame, not design limits. The plain lesson: four idle cards still turn roughly 141 W into heat, and one card's memory was already at 51 °C with no workload. An open frame alone does not cool these cards.

## Practical policies

| Mode | Measured policy | Use |
|---|---:|---|
| Quiet / idle | 125 W | Administration, downloads, idle serving |
| Benchmark | 180 W | Current reproducible performance runs with blower airflow |
| Higher power | Untested here | Only after a cooling and PSU margin study |

The power limit is a ceiling, not guaranteed consumption. Record actual board power during every result.

Set and verify an authorized limit:

```bash
sudo nvidia-smi --persistence-mode=1
sudo nvidia-smi --power-limit=125
nvidia-smi --query-gpu=index,power.draw,power.limit,temperature.gpu,temperature.memory --format=csv
```

An older service or startup script may overwrite the desired limit after boot. Always verify the live value rather than trusting a service's `active (exited)` state.

## Workload peaks to compare against

Sustained single-card load peaks are known from club measurements, but that evidence revision contains prohibited infrastructure identifiers, so the numbers and their links are withheld here pending the owner-approved history repair and sanitized re-pin already tracked in the benchmark registry. Four-card load peaks are untested; do not extrapolate one-card thermals to the full rig. Once a sanitized Qwen pin lands, this section gets the measured core/memory peaks back.

## Temperature policy

- Stop at **80 °C core**.
- Stop at **85 °C memory**.
- Investigate rising idle temperature; a passive card can retain heat even at low utilization.
- Keep the blower running during real workloads until measured data proves another airflow setup is safe.

A large external fan can help, but it should be validated by inlet/exhaust orientation and sustained memory-temperature data. Unplugging a high-static-pressure blower without an equivalent ducted replacement is not a safe default.

## Power-brake signal

Check whether the card is asserting hardware power-brake slowdown:

```bash
nvidia-smi -q | grep -A1 'HW Power Brake Slowdown'
```

Only investigate riser signal modifications when this reports `Active` and the power/riser topology supports that diagnosis. Do not tape connector pins merely because performance is low; our measured cards reported `Not Active`.

## PSU lessons

- Size for CPU, motherboard, fans, transient margin, and all GPU limits.
- Use the modular cables supplied for that exact PSU model.
- Avoid splitters and loose adapters.
- Keep riser and GPU auxiliary power domains intentional.
- A repeating standby/power-button flash can be PSU protection, cabling, board power, or a short; isolate to motherboard + CPU + one RAM configuration before blaming a GPU.
