# Cooling and power

CMP 170HX cards are passive server accelerators. An open case and a room fan are not automatically enough: air must be forced through the heatsink fins, and exhaust must not recirculate into the intake.

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
