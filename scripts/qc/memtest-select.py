#!/usr/bin/env python3
"""memtest-select.py - PTY wrapper to drive memtest_vulkan v0.5.0 non-interactively.

The upstream binary has no CLI flags: device selection is an interactive
prompt read from a real TTY (piped stdin is ignored and it silently always
tests the first-listed device). This wrapper allocates a PTY, parses the
printed device list to find the Vulkan index matching a given PCI bus id,
types that index into the prompt, lets the test run until it stops on its
own or the deadline elapses, then sends a graceful SIGTERM (the binary
prints a PASS/FAIL summary on SIGTERM) followed by SIGKILL as a backstop.

Usage: memtest-select.py <bus_id e.g. 0000:01:00.0> <max_seconds> <log_path>
Exit 0 on a clean "no any errors ... PASSed"/"testing PASSed" summary with
the correct device selected; exit 1 otherwise (error text, wrong device
confirmed, or no pass summary observed before the deadline).
"""
import os
import pty
import re
import select
import subprocess
import sys
import time

MEMTEST_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memtest_vulkan", "memtest_vulkan")


def norm_bus(b):
    # nvidia-smi gives "0000:01:00.0"; the tool prints "0x01:00"
    parts = b.split(":")
    return parts[-2].lstrip("0") or "0", parts[-1].split(".")[0]


def main():
    bus_id, max_seconds, log_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    want_bus, _ = norm_bus(bus_id)

    master, slave = pty.openpty()
    pid = os.fork()
    if pid == 0:
        os.setsid()
        os.dup2(slave, 0)
        os.dup2(slave, 1)
        os.dup2(slave, 2)
        os.close(master)
        os.execv(MEMTEST_BIN, [MEMTEST_BIN])
        os._exit(127)

    os.close(slave)
    buf = b""
    selected_idx = None
    typed = False
    deadline = time.time() + max_seconds
    sigterm_sent_at = None

    while True:
        now = time.time()
        if now >= deadline:
            break
        r, _, _ = select.select([master], [], [], 0.5)
        if master in r:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            text = buf.decode(errors="replace")

            if not typed:
                # find the device index whose "Bus=0xNN:00" matches want_bus
                for m in re.finditer(r"(\d+): Bus=0x([0-9A-Fa-f]+):00\s+DevId=0x([0-9A-Fa-f]+)", text):
                    idx, bus_hex = m.group(1), m.group(2)
                    if int(bus_hex, 16) == int(want_bus, 16):
                        selected_idx = idx
                if selected_idx and "Override index to test" in text:
                    os.write(master, f"{selected_idx}\n".encode())
                    typed = True

            if "PASSed" in text or "FAILed" in text or re.search(r"\berror", text, re.I):
                # give it a moment to flush the closing countdown, then stop
                if sigterm_sent_at is None:
                    sigterm_sent_at = now
            if sigterm_sent_at is not None and now - sigterm_sent_at > 4:
                break
        # if prompt never showed after enumeration and pty went idle, bail early only at deadline

    # NOTE: on device selection the binary re-execs itself as
    # "memtest_vulkan <idx> <bytes>" (visible via `ps`), which then gets
    # reparented to init if the wrapper's original forked pid exits first.
    # os.setsid() above makes that pid the process-group leader, and the
    # re-exec'd process inherits the same pgid (it never calls setsid), so
    # signal the whole group, not just the one pid, or VRAM stays pinned.
    try:
        os.killpg(pid, 15)
    except ProcessLookupError:
        pass
    # keep draining the pty for a few seconds so the graceful-shutdown
    # PASS/FAIL summary (printed on SIGTERM) is actually captured
    drain_until = time.time() + 6
    while time.time() < drain_until:
        r, _, _ = select.select([master], [], [], 0.5)
        if master in r:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
    try:
        os.killpg(pid, 9)
    except ProcessLookupError:
        pass
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    # last-resort backstop: any memtest_vulkan process still alive after
    # this (e.g. a grandchild that escaped the group) holds VRAM forever
    try:
        subprocess.run(["pkill", "-9", "-f", MEMTEST_BIN], check=False,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    time.sleep(1)

    out = buf.decode(errors="replace")
    with open(log_path, "w") as f:
        f.write(out)

    confirmed_right_device = False
    if selected_idx:
        confirmed_right_device = f"Standard" in out and f"{selected_idx}: Bus=0x" in out.split("Override index to test")[-1][:0] if False else True
        # simpler: check the "Standard N-minute test of <idx>: Bus=..." line matches selected_idx
        m = re.search(r"test of (\d+): Bus=0x([0-9A-Fa-f]+):00", out)
        if m:
            confirmed_right_device = int(m.group(2), 16) == int(want_bus, 16)
        else:
            confirmed_right_device = False

    passed = "no any errors" in out and "PASSed" in out
    failed = ("FAILed" in out) or ("Runtime error" in out) or ("testing failed" in out)

    if passed and confirmed_right_device and not failed:
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
