# Hardware guide

## What this card is

The CMP 170HX is a passive, compute-only NVIDIA mining card. The cards tested by this project enumerate as PCI device `10de:20c2`, expose CUDA compute capability 8.0, and report 65,536 MiB after the community unlock path is working.

That makes the card interesting for memory-heavy CUDA and AI work. It does not make it equivalent to a supported A100:

| Property | Practical effect |
|---|---|
| SM80 compute capability | A large body of CUDA software can compile for the card |
| 64 GiB HBM exposed after patching | Models that do not fit consumer 24 GiB cards may fit |
| No NVLink | Multi-card collectives cross PCIe and can dominate runtime |
| Passive heatsink | Forced front-to-back airflow is mandatory under load |
| Compute-only device | No display output; do not configure it as a primary VGA card |
| Unsupported unlock path | Version pinning and recovery procedures are part of normal operation |

## Before buying or installing

Confirm these items first:

1. The chassis can deliver directed airflow through the heatsink, not merely move air around an open frame.
2. The PSU and cabling can supply every card without splitters or mixed modular-PSU cables.
3. The motherboard can enumerate the desired slot/riser topology and has Above 4G Decoding support.
4. The host can tolerate a full cold power cycle when PCIe recovery fails.
5. The workload benefits from capacity enough to offset slow inter-card communication.

Never mix modular PSU cables between PSU models, even when the connector fits. Keep each riser and its GPU power on a deliberate, documented power domain.

## Our four-card rig

The current test rig holds four CMP 170HX cards in an open-frame chassis with enough spacing between slots to reach every power connector. Each card gets its own cable; there are no splitters. Cables are dressed to the frame so they cannot sag into neighboring fans or block a heatsink face.

Because the cards are passive, the frame depends on one 80 mm blower pushing air through a 3D-printed duct seated over the heatsink inlet at one end of the stack. The duct feeds the fin channels; air enters at the ducted end and exhausts out the opposite side of the frame, so keep a clear lane at both ends. Don't crowd either side; these cards heat quickly when airflow stalls.

The blower speed is fixed; the cards provide no fan-speed control, so treat the blower as always-on hardware rather than something the driver manages. Its RPM is unmeasured and ambient temperature was not recorded during the current idle snapshot, which is one reason these numbers describe this frame and not a product.

## First inventory

With no workload running:

```bash
lspci -nn | grep -iE 'NVIDIA|3D controller'
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,power.limit,temperature.gpu --format=csv
nvidia-smi -q | grep -A1 'HW Power Brake Slowdown'
```

Save a redacted baseline outside the public repository. A card missing from `lspci` is a hardware/firmware/power/enumeration problem; reinstalling a guest driver will not make a non-enumerated PCI device appear.

## PCIe link status

**Measured 2026-09-02, on the four-card test node.** Host-side `lspci -vv` on
the hypervisor shows every CMP 170HX card advertising `LnkCap: Speed
2.5GT/s, Width x16`. PCIe Gen1 is the card's advertised maximum, so a Gen1
link seen in a guest is a card/vBIOS capability, not a hypervisor
passthrough bug. One card trains at x8 instead of the advertised x16. On this
board, a fully populated seven-slot dual-socket layout shares lanes between
neighboring slots, so a neighbor's card occupying the adjacent slot drops
that card to x8 — this is expected board topology, not a riser fault or a
passthrough bug (an earlier note in this section called it a riser
suspicion; the 2026-09-02 health gate traced it to lane sharing instead).
For comparison, consumer GeForce cards on the same host show x1 width at
2.5GT/s at idle; that is normal ASPM power-saving behavior, not a fault.

## Known-good measured configuration

**Measured inventory:** four cards enumerated in one Ubuntu guest on 2026-08-30 and exposed 65,536 MiB each, for 256 GiB aggregate. Four-card workload performance remains untested.

**Earlier bring-up snapshot (pre-duct arrangement):** at idle with a 180 W cap during initial bring-up, before the current open-frame duct described above was in place, the four cards reported 31–32 °C core temperatures, 35–45 °C memory temperatures, 32–36 W per card, and 0% GPU utilization. This is a different setup state and a different measurement from the current rig; do not blend the ranges. The current-arrangement idle table lives in [Cooling and power](COOLING-AND-POWER.md#idle-heat-is-real-heat).

**Measured:** three cards passed CUDA enumeration, a 16 GiB cross-window write/read smoke test, an SM kernel, and post-load health checks under Ubuntu 22.04, Linux 6.8, driver 610.43.03, and the pinned unlocker described in [Installation](INSTALLATION.md).

This validates that exact configuration. Other card revisions and software versions remain untested until evidence is added.
