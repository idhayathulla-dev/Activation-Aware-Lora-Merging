#!/usr/bin/env python3
"""
Environment Preflight & Hardware Diagnostics Script
Audits Python, PyTorch, CUDA, RAM, Disk Space, and critical dependency versions.
"""

import sys
import os
import json
import shutil
import platform
from pathlib import Path

def get_system_info():
    report = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "architecture": platform.machine(),
        "processor": platform.processor(),
    }

    # RAM
    try:
        import psutil
        vm = psutil.virtual_memory()
        report["ram_total_gb"] = round(vm.total / (1024 ** 3), 2)
        report["ram_available_gb"] = round(vm.available / (1024 ** 3), 2)
        report["ram_percent_used"] = vm.percent
    except ImportError:
        report["ram_total_gb"] = "psutil not installed"
        report["ram_available_gb"] = "psutil not installed"
        report["ram_percent_used"] = "N/A"

    # Disk Space
    try:
        total, used, free = shutil.disk_usage(".")
        report["disk_total_gb"] = round(total / (1024 ** 3), 2)
        report["disk_used_gb"] = round(used / (1024 ** 3), 2)
        report["disk_free_gb"] = round(free / (1024 ** 3), 2)
    except Exception as e:
        report["disk_error"] = str(e)

    # PyTorch and CUDA
    try:
        import torch
        report["pytorch_version"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["cuda_version"] = torch.version.cuda
            report["device_count"] = torch.cuda.device_count()
            report["gpu_name"] = torch.cuda.get_device_name(0)
            report["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
            )
        else:
            report["cuda_version"] = None
            report["gpu_name"] = "None (CPU only)"
            report["gpu_memory_gb"] = 0.0
    except ImportError:
        report["pytorch_version"] = "Not Installed"
        report["cuda_available"] = False
        report["gpu_name"] = "PyTorch not installed"

    # Key packages
    packages_to_check = [
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("yaml", "PyYAML"),
        ("pytest", "pytest"),
        ("safetensors", "safetensors"),
    ]
    installed_pkgs = {}
    for mod_name, label in packages_to_check:
        try:
            mod = __import__(mod_name)
            installed_pkgs[label] = getattr(mod, "__version__", "installed")
        except ImportError:
            installed_pkgs[label] = "MISSING"

    report["installed_packages"] = installed_pkgs
    return report

def main():
    print("=" * 60)
    print(" ACTIVATION-AWARE LORA MERGING: PREFLIGHT AUDIT ")
    print("=" * 60)
    info = get_system_info()
    
    print(f"OS Platform          : {info['platform']}")
    print(f"Python Executable    : {info['python_executable']}")
    print(f"Python Version       : {info['python_version'].split()[0]}")
    print(f"Processor            : {info['processor']}")
    print(f"RAM (Total / Avail)  : {info['ram_total_gb']} GB / {info['ram_available_gb']} GB")
    print(f"Disk Free            : {info.get('disk_free_gb', 'N/A')} GB / {info.get('disk_total_gb', 'N/A')} GB")
    print(f"PyTorch Version      : {info['pytorch_version']}")
    print(f"CUDA Available       : {info['cuda_available']}")
    print(f"GPU Hardware         : {info.get('gpu_name', 'None')}")
    print("-" * 60)
    print("Core Dependencies Status:")
    for pkg, ver in info["installed_packages"].items():
        status = "OK" if ver != "MISSING" else "NOT INSTALLED"
        print(f"  - {pkg:<16} : {ver:<15} [{status}]")
    print("=" * 60)

    # Save to report directory inside project root
    project_root = Path(__file__).resolve().parent.parent
    reports_dir = project_root / "results" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "preflight_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Preflight report saved to: {report_file}")

    if not info.get("cuda_available", False):
        print("\n[NOTE] Local environment is CPU-only. Heavy training and fine-tuning")
        print("       should be executed on Google Colab or a GPU server using the")
        print("       provided standalone notebook. Local CPU is configured for unit")
        print("       tests, synthetic matrix verification, and pipeline smoke tests.")

if __name__ == "__main__":
    main()
