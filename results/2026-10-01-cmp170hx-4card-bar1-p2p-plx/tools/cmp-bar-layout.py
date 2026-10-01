#!/usr/bin/env python3
"""Program a CMP 170HX PCIe layout that fits a full 64 GB BAR1 per card.

Run with no driver bound to the GPUs, then kexec so the kernel adopts the
layout as firmware assignments. Each card's 32 MB BAR3 sits directly below
its 64 GB-aligned BAR1, so two cards fit behind one PLX switch (the kernel's
own sizing places BAR3 after BAR1 and runs out of room for the second card).
"""
import os
import subprocess
import sys

G = 1 << 30
M = 1 << 20
BAR1 = 64 * G
BAR3 = 32 * M

# card BDF -> (downstream port, 64 GB-aligned BAR1 base)
CARDS = {
    "84:00.0": ("83:08.0", 0x31000000000),
    "85:00.0": ("83:10.0", 0x33000000000),
    "88:00.0": ("87:08.0", 0x35000000000),
    "89:00.0": ("87:10.0", 0x37000000000),
}
# bridges that must cover a set of cards
UPSTREAM = {
    "82:00.0": ["84:00.0", "85:00.0"],
    "80:02.0": ["84:00.0", "85:00.0"],
    "86:00.0": ["88:00.0", "89:00.0"],
    "80:03.0": ["88:00.0", "89:00.0"],
}
DRY = "--dry-run" in sys.argv
UNLOCK = "--unlock" in sys.argv
CMP_IDS = ("0x10de", "0x20c2")


def sysfs(bdf, attr):
    with open(f"/sys/bus/pci/devices/0000:{bdf}/{attr}") as f:
        return f.read().strip()


def refuse(msg):
    """Exit 2: nothing was written (cmp-p2p-boot falls back to safe mode without a kexec)."""
    print(f"{msg}; layout not applied", file=sys.stderr)
    sys.exit(2)


def check_topology():
    """Refuse to touch anything unless the cards sit exactly where the layout expects."""
    for card, (port, _) in CARDS.items():
        if not os.path.exists(f"/sys/bus/pci/devices/0000:{card}"):
            refuse(f"{card} not present")
        if (sysfs(card, "vendor"), sysfs(card, "device")) != CMP_IDS:
            refuse(f"{card} is not a CMP 170HX")
        if os.path.exists(f"/sys/bus/pci/devices/0000:{card}/driver"):
            refuse(f"{card} already has a driver bound")
        parent = os.path.basename(os.path.dirname(os.path.realpath(f"/sys/bus/pci/devices/0000:{card}")))
        if parent != f"0000:{port}":
            refuse(f"{card} sits behind {parent}, expected {port}")


def setpci(bdf, reg, val):
    arg = f"{reg}={val}"
    print(f"setpci -s {bdf} {arg}")
    if not DRY:
        subprocess.run(["setpci", "-s", bdf, arg], check=True)


def card_range(bar1):
    return bar1 - BAR3, bar1 + BAR1 - 1


def set_window(bdf, base, limit):
    assert base % M == 0 and (limit + 1) % M == 0
    setpci(bdf, "0x24.w", f"{((base >> 16) & 0xfff0) | 1:04x}")
    setpci(bdf, "0x26.w", f"{((limit >> 16) & 0xfff0) | 1:04x}")
    setpci(bdf, "0x28.l", f"{base >> 32:08x}")
    setpci(bdf, "0x2c.l", f"{limit >> 32:08x}")


def set_bar64(bdf, reg_lo, addr):
    setpci(bdf, f"{reg_lo:#x}.l", f"{(addr & 0xffffffff) | 0xc:08x}")
    setpci(bdf, f"{reg_lo + 4:#x}.l", f"{addr >> 32:08x}")


check_topology()

for card, (port, bar1) in CARDS.items():
    assert bar1 % BAR1 == 0, card
    setpci(card, "COMMAND", "0000:0006")  # memory/bus-master decode off
    if UNLOCK:
        # Same XVE writes the driver's nv_cmp_unlock_bar1_resize() does, through
        # config space: open the ReBAR capability up to 64 GB, then select it.
        setpci(card, "0x724.l", "00000030")
        setpci(card, "0xdcc.l", "8000000a")
        setpci(card, "0xbc0.w", "1000:3f00")  # ReBAR ctrl BAR1 size field = 16 (64 GB)
    set_bar64(card, 0x14, bar1)           # BAR1
    set_bar64(card, 0x1c, bar1 - BAR3)    # BAR3 just below BAR1
    set_window(port, *card_range(bar1))

for bridge, cards in UPSTREAM.items():
    ranges = [card_range(CARDS[c][1]) for c in cards]
    set_window(bridge, min(r[0] for r in ranges), max(r[1] for r in ranges))
