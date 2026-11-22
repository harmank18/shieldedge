"""Device identification used to label every benchmark with the hardware it ran on."""

import os
import platform
import re


def device_name() -> str:
    """'Raspberry Pi 5 Model B Rev 1.0' on a Pi, CPU model name elsewhere."""
    try:
        with open("/proc/device-tree/model") as f:
            return f.read().strip("\x00\n ")
    except OSError:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def device_slug() -> str:
    """Short file-name-safe identifier, e.g. 'raspberry-pi-5' or 'x86_64-i5-1235u'."""
    name = device_name().lower()
    if "raspberry pi 5" in name:
        return "raspberry-pi-5"
    if "raspberry pi" in name:
        return re.sub(r"[^a-z0-9]+", "-", name.split(" rev")[0]).strip("-")
    m = re.search(r"(i[3579]-\w+|ryzen \d \w+|apple m\d\w*)", name)
    return f"{platform.machine()}-{m.group(1).replace(' ', '-')}" if m else platform.machine()


def device_info() -> dict:
    total = None
    try:
        import psutil

        total = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    return {
        "device": device_name(),
        "arch": platform.machine(),
        "cpus": os.cpu_count(),
        "ram_gb": total,
        "os": platform.platform(),
        "python": platform.python_version(),
    }
