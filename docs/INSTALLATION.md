# Installation: Ubuntu guest on Proxmox

This is a reconstruction of a measured working setup with private infrastructure identifiers removed. It is not an officially supported NVIDIA procedure.

## Known-good boundary

| Layer | Verified value |
|---|---|
| Guest | Ubuntu Server 22.04 |
| Kernel | Linux 6.8 |
| VM machine | Q35 |
| VM firmware | SeaBIOS |
| GPU role | PCI passthrough, compute-only; no `x-vga=1` |
| NVIDIA driver | 610.43.03, open kernel module |
| Unlocker | `amoghmunikote/cmpunlocker` commit `7019bc2b407ebf776d1b0836a0569a3dec52c4f5` |
| Unlock profile | `8gb`, patches 0001–0006 |

OVMF/UEFI repeatedly produced `RmInitAdapter` and SEC2 initialization failures in our Proxmox VM. The same cards initialized under SeaBIOS. The experimental Gen2 path also caused guest/QEMU hangs and is excluded from this baseline.

## 1. Prepare the VM

Before changing an existing UEFI guest to SeaBIOS, make sure it can boot in BIOS mode. On the measured Ubuntu disk:

```bash
sudo grub-install --target=i386-pc --recheck /dev/sda
sudo update-grub
```

Disk layouts differ. Confirm the target disk and the presence of a BIOS boot partition before running `grub-install`; choosing the wrong disk can make the guest unbootable.

Set Q35 + SeaBIOS, use serial or a separate display device for the console, and pass each CMP device as PCIe hardware. Do not set `x-vga=1` on a compute-only CMP card.

Secure Boot must either be disabled or correctly configured to trust the patched kernel modules. A module that builds successfully can still be rejected at load time.

## 2. Install the exact driver

Obtain the 610.43.03 Linux runfile from NVIDIA and verify its checksum before use. Then install the open kernel module:

```bash
sudo sh NVIDIA-Linux-x86_64-610.43.03.run \
  --silent \
  --kernel-module-type=open \
  --dkms \
  --no-x-check \
  --disable-nouveau \
  --rebuild-initramfs
```

Review installer output and confirm the running kernel has matching headers.

## 3. Pin and install the unlocker

Review the third-party code before running it as root:

```bash
git clone https://github.com/amoghmunikote/cmpunlocker cmpunlocker-v01
cd cmpunlocker-v01
git checkout 7019bc2b407ebf776d1b0836a0569a3dec52c4f5
sudo ./install.sh --profile=8gb
```

Do not enable the experimental Gen2 patch/service on this VM baseline.

## 4. Eliminate duplicate modules

A stock DKMS copy under `updates/dkms` can win module resolution over the patched copy under `updates/cmpunlocker`. The observed symptom was an 8 GiB memory map despite a successful patch install.

Inspect before removing anything:

```bash
modinfo -n nvidia
dkms status
```

If the duplicate NVIDIA 610.43.03 DKMS entry is present and the patched module is intended, the measured repair was:

```bash
sudo dkms remove nvidia/610.43.03 --all
sudo depmod -a
sudo update-initramfs -u
modinfo -n nvidia
```

`modinfo -n nvidia` must resolve to `updates/cmpunlocker/nvidia.ko`. Then fully shut down and start the guest; a warm module reload is not equivalent to resetting the PCI device.

## 5. Verify before load

```bash
nvidia-smi --query-gpu=index,pci.bus_id,memory.total,power.limit --format=csv
modinfo nvidia | grep -E '^(filename|version):'
sudo dmesg --level=err,warn | grep -Ei 'NVRM|Xid|fallen off|RmInitAdapter|SEC2|AER'
```

Continue with [QC and acceptance testing](QC.md). Do not treat `nvidia-smi` enumeration alone as proof that the upper memory region and SMs are healthy.

## Primary sources

- [NVIDIA 610.43.03 installer documentation](https://download.nvidia.com/XFree86/Linux-x86_64/610.43.03/README/installdriver.html)
- [NVIDIA open kernel-module documentation](https://download.nvidia.com/XFree86/Linux-x86_64/610.43.03/README/kernel_open.html)
- [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker)
- [Community Proxmox procedure](https://github.com/Consensus-Protocol/cmp170hx/blob/main/docs/procedures/install.md)
- [PixelML QC source commit](https://github.com/PixelML/nuoa/commit/b8c68e22b81d0b4777c9109d988438b4dbe985bf)
