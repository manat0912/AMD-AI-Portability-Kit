#!/usr/bin/env python3
"""AMD AI portability kit.

Copy this file into the root of a Python/CUDA/ONNX project and run it there.
It inventories accelerator lock-in, selects a realistic AMD-compatible runtime,
creates a reviewable migration bundle, and can make only narrow, reversible
Python changes when --apply is supplied.

Examples
--------
python portable_ai.py scan .
python portable_ai.py prepare . --target directml
python portable_ai.py prepare . --target directml --apply --install
python portable_ai.py prepare . --target rocm --rocm-index-url URL --install

This is deliberately conservative: it cannot turn a precompiled CUDA wheel,
TensorRT engine, or proprietary executable into AMD code.  Those cases are
reported as blockers rather than being "converted" into an app that fails at
runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import platform as host_platform
import re
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


VERSION = "2.0.0"
MAX_TEXT_BYTES = 4 * 1024 * 1024
MAX_FILES = 15000
SKIP_DIRS = {
    ".git", ".hg", ".svn", ".ai-portability", "node_modules", "venv",
    ".venv", "env", "__pycache__", ".mypy_cache", ".pytest_cache",
    "build", "dist", ".tox", ".eggs",
}
TEXT_SUFFIXES = {
    ".py", ".pyi", ".cu", ".cuh", ".c", ".cc", ".cpp", ".cxx", ".h",
    ".hh", ".hpp", ".cmake", ".txt", ".toml", ".yaml", ".yml", ".json",
    ".js", ".mjs", ".ts", ".tsx", ".sh", ".bat", ".ps1", ".md", ".rst",
}
TEXT_NAMES = {
    "CMakeLists.txt", "Makefile", "Dockerfile", "requirements.txt",
    "environment.yml", "environment.yaml", "setup.py", "setup.cfg",
    "pyproject.toml", "package.json",
}
CUDA_SUFFIXES = {".cu", ".cuh"}
NATIVE_BINARY_SUFFIXES = {".dll", ".so", ".dylib", ".pyd", ".exe", ".a", ".lib"}

# These are intentionally source-level signals, not an attempt to identify a
# project from its name.  Keeping the evidence makes the report auditable.
SIGNALS = (
    ("torch_cuda", r"\btorch\.cuda\b", "warning", "PyTorch CUDA API"),
    ("cuda_extension", r"\b(?:CUDAExtension|CppExtension|load_inline|torch\.utils\.cpp_extension)\b", "blocker", "PyTorch native/CUDA extension"),
    ("cuda_source", r"(?:#\s*include\s*[<\"]cuda|\b(?:cudaMalloc|cudaMemcpy|__global__|__device__|nvcc)\b)", "warning", "CUDA source or API"),
    ("cudnn", r"\b(?:cudnn|cuDNN|CUDNN_)\b", "warning", "cuDNN dependency"),
    ("cublas", r"\b(?:cublas|cuBLAS)\b", "warning", "cuBLAS dependency"),
    ("nccl", r"\b(?:nccl|NCCL_)\b", "warning", "NCCL dependency"),
    ("tensorrt", r"(?:\b(?:import|from)\s+tensorrt\b|(?m:^\s*tensorrt(?:\s*(?:[<>=!~]|\[)|\s*$))|\btrtexec\b)", "blocker", "TensorRT dependency"),
    ("triton", r"\b(?:triton\.language|triton\.jit|@triton\.jit)\b", "blocker", "Triton kernel"),
    ("flash_attention", r"\b(?:flash[_-]?attn|flash_attn|xformers|bitsandbytes)\b", "warning", "CUDA-focused performance extension"),
    ("cupy", r"\b(?:cupy|pycuda)\b", "warning", "CUDA Python package"),
    ("nvidia_tools", r"\b(?:nvidia-smi|pynvml|NVML|CUDA_VISIBLE_DEVICES)\b", "warning", "NVIDIA runtime tooling"),
    ("onnx", r"\b(?:onnxruntime|InferenceSession|CUDAExecutionProvider|TensorrtExecutionProvider)\b", "warning", "ONNX Runtime provider"),
    ("tensorflow", r"\b(?:tensorflow|tf\.config\.experimental\.list_physical_devices)\b", "info", "TensorFlow"),
    ("jax", r"\b(?:jax|jaxlib)\b", "info", "JAX"),
)

CUDA_TO_HIP = {
    "cuda_runtime.h": "hip/hip_runtime.h",
    "cuda.h": "hip/hip_runtime.h",
    "cuda_runtime_api.h": "hip/hip_runtime_api.h",
    "cuda_fp16.h": "hip/hip_fp16.h",
    "cublas_v2.h": "rocblas/rocblas.h",
    "cudnn.h": "miopen/miopen.h",
    "curand.h": "rocrand/rocrand.h",
    "cusparse.h": "hipsparse/hipsparse.h",
    "nccl.h": "rccl/rccl.h",
    "cudaMalloc": "hipMalloc",
    "cudaFree": "hipFree",
    "cudaMemcpy": "hipMemcpy",
    "cudaMemset": "hipMemset",
    "cudaDeviceSynchronize": "hipDeviceSynchronize",
    "cudaGetLastError": "hipGetLastError",
    "cudaGetErrorString": "hipGetErrorString",
    "cudaError_t": "hipError_t",
    "cudaSuccess": "hipSuccess",
    "cudaStream_t": "hipStream_t",
    "cudaEvent_t": "hipEvent_t",
    "cudaMemcpyHostToDevice": "hipMemcpyHostToDevice",
    "cudaMemcpyDeviceToHost": "hipMemcpyDeviceToHost",
    "cudaMemcpyDeviceToDevice": "hipMemcpyDeviceToDevice",
    "nvcc": "hipcc",
}

TARGET_INFO = {
    "directml": {
        "label": "DirectML / ONNX Runtime DirectML",
        "platforms": {"windows"},
        "summary": "Broad Windows GPU compatibility. Best for Python applications without custom CUDA kernels.",
        "docs": "https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows",
    },
    "rocm": {
        "label": "PyTorch ROCm / HIP",
        "platforms": {"linux", "windows"},
        "summary": "Best path for PyTorch and source-available CUDA on supported AMD hardware.",
        "docs": "https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/prerequisites.html",
    },
    "migraphx": {
        "label": "ONNX Runtime MIGraphX",
        "platforms": {"linux"},
        "summary": "Current AMD ONNX Runtime accelerator path on Linux ROCm.",
        "docs": "https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html",
    },
    "onnx_directml": {
        "label": "ONNX Runtime DirectML",
        "platforms": {"windows"},
        "summary": "Windows ONNX inference path; use the DirectML provider with CPU fallback.",
        "docs": "https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html",
    },
    "onnx_cpu": {
        "label": "ONNX Runtime CPU",
        "platforms": {"linux", "windows", "darwin"},
        "summary": "Portable fallback when a GPU runtime cannot be used.",
        "docs": "https://onnxruntime.ai/docs/install/",
    },
}


def _normalise_platform(value: str) -> str:
    value = (value or "auto").lower()
    if value == "auto":
        value = host_platform.system().lower()
    aliases = {"win32": "windows", "cygwin": "windows", "linux2": "linux", "macos": "darwin", "osx": "darwin"}
    return aliases.get(value, value)


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES


def _read_text(path: Path) -> Optional[str]:
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError):
        return None


def _finding(identifier: str, severity: str, title: str, detail: str, paths: Iterable[str]) -> Dict[str, object]:
    return {
        "id": identifier,
        "severity": severity,
        "title": title,
        "detail": detail,
        "paths": sorted(set(paths))[:20],
    }


def scan_project(project: str, platform_name: str = "auto") -> Dict[str, object]:
    """Inventory accelerator dependencies without changing the project."""
    root = Path(project).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("Project directory does not exist: {0}".format(root))

    platform_name = _normalise_platform(platform_name)
    hits: Dict[str, List[str]] = {signal[0]: [] for signal in SIGNALS}
    compiled = [(identifier, re.compile(pattern, re.IGNORECASE), severity, title) for identifier, pattern, severity, title in SIGNALS]
    files_scanned = 0
    text_files = 0
    cuda_files: List[str] = []
    binary_files: List[str] = []
    manifests: List[str] = []
    oversized: List[str] = []
    python_files: List[str] = []
    onnx_models: List[str] = []

    for current, dirs, names in os.walk(str(root)):
        dirs[:] = [directory for directory in dirs if directory not in SKIP_DIRS and not directory.startswith(".git")]
        for name in names:
            files_scanned += 1
            if files_scanned > MAX_FILES:
                break
            path = Path(current) / name
            relative = path.relative_to(root).as_posix()
            suffix = path.suffix.lower()
            # The portability kit is frequently copied into a project.  Its
            # own catalogue must never become evidence about that project.
            if path.parent == root and path.name in {"portable_ai.py", "amd_portability.py"}:
                continue
            if suffix in CUDA_SUFFIXES:
                cuda_files.append(relative)
            if suffix in NATIVE_BINARY_SUFFIXES:
                binary_files.append(relative)
            if suffix in {".py", ".pyi"}:
                python_files.append(relative)
            if suffix == ".onnx":
                onnx_models.append(relative)
            if path.name in TEXT_NAMES or path.name in {"pyproject.toml", "package.json"}:
                manifests.append(relative)
            if not _is_text_file(path):
                continue
            try:
                if path.stat().st_size > MAX_TEXT_BYTES:
                    oversized.append(relative)
                    continue
            except OSError:
                continue
            content = _read_text(path)
            if content is None:
                continue
            text_files += 1
            # Documentation commonly names CUDA, TensorRT, and competing
            # runtimes; that is not project lock-in. Search implementation and
            # dependency/build manifests, not README prose.
            if suffix in {".md", ".rst"}:
                continue
            for identifier, expression, _severity, _title in compiled:
                if expression.search(content):
                    hits[identifier].append(relative)
        if files_scanned > MAX_FILES:
            break

    findings: List[Dict[str, object]] = []
    for identifier, _expression, severity, title in compiled:
        if hits[identifier]:
            detail = "Detected in {0} file(s).".format(len(hits[identifier]))
            if identifier in {"cuda_extension", "triton", "tensorrt"}:
                detail += " This normally needs an upstream AMD build, a HIP port, or an ONNX/CPU replacement; runtime package installation alone is insufficient."
            elif identifier == "onnx":
                detail += " Provider selection can be made portable without changing the model format."
            findings.append(_finding(identifier, severity, title, detail, hits[identifier]))

    if cuda_files:
        findings.append(_finding(
            "cuda_files", "warning", "CUDA source files", "Source is available, so HIP conversion is possible but must be compiled and tested with hipcc.", cuda_files
        ))
    if binary_files:
        findings.append(_finding(
            "native_binaries", "blocker", "Precompiled native binaries", "Binary CUDA/NVIDIA components cannot be safely rewritten. Obtain an AMD build, rebuild from source, or replace the component.", binary_files
        ))
    if oversized:
        findings.append(_finding(
            "oversized_files", "info", "Large files skipped", "Files larger than {0} MiB were not searched to keep scanning safe and fast.".format(MAX_TEXT_BYTES // (1024 * 1024)), oversized
        ))
    if files_scanned > MAX_FILES:
        findings.append(_finding(
            "scan_limit", "warning", "Scan limit reached", "Only the first {0} files were scanned. Narrow the project root for a complete audit.".format(MAX_FILES), []
        ))

    ids = {item["id"] for item in findings}
    has_python = bool(python_files)
    has_onnx = "onnx" in ids or bool(onnx_models)
    has_cuda = bool(cuda_files) or any(identifier in ids for identifier in ("torch_cuda", "cuda_source", "cuda_extension"))

    candidates = recommend_targets(platform_name, has_python, has_onnx, has_cuda, ids)
    selected = candidates[0]["id"] if candidates else "onnx_cpu"
    return {
        "tool_version": VERSION,
        "project": str(root),
        "platform": platform_name,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "files_scanned": files_scanned,
        "text_files_scanned": text_files,
        "manifests": sorted(set(manifests)),
        "cuda_files": sorted(cuda_files),
        "native_binaries": sorted(binary_files),
        "has_python": has_python,
        "has_onnx": has_onnx,
        "has_cuda": has_cuda,
        "findings": findings,
        "candidates": candidates,
        "recommended_target": selected,
    }


def recommend_targets(platform_name: str, has_python: bool, has_onnx: bool, has_cuda: bool, finding_ids: Iterable[str]) -> List[Dict[str, object]]:
    """Rank known, supportable paths; never claim a binary can be ported."""
    finding_ids = set(finding_ids)
    candidates: List[Tuple[str, int, str]] = []
    if platform_name == "windows":
        if has_onnx:
            candidates.append(("onnx_directml", 95, "Existing ONNX inference can use DirectML with CPU fallback."))
        if has_python:
            candidates.append(("directml", 88, "Python app detected; DirectML avoids a full CUDA source port when its operators are sufficient."))
        if has_cuda and not ({"cuda_extension", "triton", "tensorrt", "native_binaries"} & finding_ids):
            candidates.append(("rocm", 62, "Source-level HIP port may work on the supported Windows ROCm/PyTorch matrix."))
    elif platform_name == "linux":
        if has_onnx:
            candidates.append(("migraphx", 96, "ONNX Runtime MIGraphX is AMD's current Linux ROCm execution-provider path."))
        if has_python or has_cuda:
            candidates.append(("rocm", 92, "ROCm/HIP is the primary Linux route for PyTorch and source-available CUDA."))
    if has_onnx or not candidates:
        candidates.append(("onnx_cpu", 30, "CPU fallback preserves portability where no compatible GPU route is available."))

    unique: Dict[str, Tuple[int, str]] = {}
    for identifier, score, reason in candidates:
        prior = unique.get(identifier)
        if prior is None or score > prior[0]:
            unique[identifier] = (score, reason)
    result = []
    for identifier, (score, reason) in sorted(unique.items(), key=lambda item: item[1][0], reverse=True):
        # TARGET_INFO uses a set internally for membership tests; reports must
        # remain JSON serialisable so that a bundle is portable and auditable.
        info = dict(TARGET_INFO[identifier])
        info["platforms"] = sorted(info["platforms"])
        result.append({"id": identifier, "score": score, "reason": reason, **info})
    return result


def choose_target(report: Dict[str, object], target: str) -> str:
    target = target.lower()
    if target == "auto":
        return str(report["recommended_target"])
    if target not in TARGET_INFO:
        raise ValueError("Unknown target '{0}'. Choose one of: auto, {1}".format(target, ", ".join(sorted(TARGET_INFO))))
    platform_name = str(report["platform"])
    if platform_name not in TARGET_INFO[target]["platforms"]:
        raise ValueError("Target '{0}' is not supported by this tool on {1}.".format(target, platform_name))
    return target


def installation_commands(target: str, python_executable: str, rocm_index_url: Optional[str], migraphx_wheel: Optional[str]) -> Tuple[List[List[str]], List[str]]:
    """Return executable commands and explicit prerequisites for a target."""
    commands: List[List[str]] = []
    prerequisites: List[str] = []
    pip = [python_executable, "-m", "pip", "install", "--upgrade"]
    if target == "directml":
        commands.append(pip + ["torch-directml"])
    elif target == "onnx_directml":
        commands.append(pip + ["onnxruntime-directml"])
    elif target == "onnx_cpu":
        commands.append(pip + ["onnxruntime"])
    elif target == "rocm":
        if rocm_index_url:
            commands.append(pip + ["torch", "torchvision", "--index-url", rocm_index_url])
        else:
            prerequisites.append("Set --rocm-index-url to the exact official wheel index matching your ROCm release, Python version, OS, and GPU support matrix. The script will not guess a wheel ABI.")
    elif target == "migraphx":
        if migraphx_wheel:
            commands.append(pip + [migraphx_wheel])
        else:
            prerequisites.append("Set --migraphx-wheel to the exact official ONNX Runtime MIGraphX wheel URL/path matching your ROCm release and Python ABI. The script will not install a mismatched binary.")
    return commands, prerequisites


RUNTIME_HELPER = r'''"""Generated by portable_ai.py.  Keep this module project-local and import it explicitly."""
from __future__ import annotations

import os

TARGET = os.environ.get("AMD_PORTABILITY_TARGET", "{target}").lower()


def _torch():
    import torch
    return torch


def device():
    """Return a torch-compatible device for ROCm, DirectML, or CPU."""
    torch = _torch()
    if TARGET == "directml":
        try:
            import torch_directml
            return torch_directml.device()
        except ImportError as exc:
            raise RuntimeError("DirectML was selected but torch-directml is not installed.") from exc
    # ROCm intentionally uses torch.device('cuda'): that is PyTorch's public
    # device spelling for both CUDA and ROCm builds.
    if TARGET == "rocm" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_accelerated():
    value = device()
    return str(value) != "cpu"


def onnx_providers():
    """Return only providers present in the installed ONNX Runtime build."""
    import onnxruntime as ort
    available = set(ort.get_available_providers())
    preferences = {{
        "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
        "onnx_directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
        "migraphx": ["MIGraphXExecutionProvider", "CPUExecutionProvider"],
        "rocm": ["MIGraphXExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"],
    }}.get(TARGET, ["CPUExecutionProvider"])
    selected = [provider for provider in preferences if provider in available]
    return selected or ["CPUExecutionProvider"]


def inference_session(model_path, **kwargs):
    import onnxruntime as ort
    kwargs.setdefault("providers", onnx_providers())
    return ort.InferenceSession(model_path, **kwargs)
'''


def _insertion_line(source: str) -> int:
    """Choose a stable place after shebang, encoding, module docstring and imports."""
    lines = source.splitlines(keepends=True)
    index = 0
    if lines and lines[0].startswith("#!"):
        index = 1
    if index < len(lines) and re.match(r"#.*coding[:=]", lines[index]):
        index += 1
    while index < len(lines) and (not lines[index].strip() or lines[index].lstrip().startswith("#")):
        index += 1
    if index < len(lines) and re.match(r"\s*(?:'''|\"\"\")", lines[index]):
        marker = lines[index].lstrip()[:3]
        index += 1
        while index < len(lines) and marker not in lines[index]:
            index += 1
        index += 1
    last_import = index
    for position in range(index, len(lines)):
        if re.match(r"\s*(?:from\s+\S+\s+import|import\s+\S+)", lines[position]):
            last_import = position + 1
        elif lines[position].strip() and last_import > index:
            break
    return last_import


def patch_python_source(source: str) -> Tuple[str, List[str]]:
    """Apply only simple, reviewable device/provider substitutions.

    CUDA kernel code, dynamic strings and complex InferenceSession calls are
    deliberately left untouched and appear in the report as manual work.
    """
    changed: List[str] = []
    rewritten = source
    substitutions = (
        (r"torch\.device\(\s*(['\"])cuda(?::\d+)?\1\s*\)", "amd_device()", "torch.device('cuda')"),
        (r"\.to\(\s*(['\"])cuda(?::\d+)?\1\s*\)", ".to(amd_device())", ".to('cuda')"),
        (r"\.cuda\(\s*\)", ".to(amd_device())", ".cuda()"),
        (r"torch\.cuda\.is_available\(\s*\)", "amd_is_accelerated()", "torch.cuda.is_available()"),
        (r"providers\s*=\s*\[\s*(['\"])CUDAExecutionProvider\1\s*,\s*(['\"])CPUExecutionProvider\2\s*\]", "providers=amd_onnx_providers()", "ONNX CUDA provider list"),
    )
    for pattern, replacement, label in substitutions:
        rewritten, count = re.subn(pattern, replacement, rewritten)
        if count:
            changed.append("{0}: {1}".format(label, count))
    if changed and "from amd_portability import" not in rewritten:
        lines = rewritten.splitlines(keepends=True)
        line = _insertion_line(rewritten)
        lines.insert(line, "from amd_portability import device as amd_device, is_accelerated as amd_is_accelerated, onnx_providers as amd_onnx_providers\n")
        rewritten = "".join(lines)
        changed.append("inserted amd_portability import")
    return rewritten, changed


def convert_cuda_source(source: str) -> Tuple[str, int]:
    """Create a limited HIP draft.  It is a review artifact, not a build guarantee."""
    output = source
    replacements = 0
    for old, new in CUDA_TO_HIP.items():
        output, count = re.subn(r"\b" + re.escape(old) + r"\b", new, output)
        replacements += count
    return output, replacements


def _write_if_changed(path: Path, content: str, force: bool = False) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def _alternative_requirements(root: Path, bundle: Path, target: str) -> Optional[Path]:
    requirements = root / "requirements.txt"
    if not requirements.is_file():
        return None
    lines = requirements.read_text(encoding="utf-8", errors="replace").splitlines()
    removed = ("torch", "torchvision", "torchaudio", "torch-directml", "onnxruntime")
    preserved = [line for line in lines if not line.strip().lower().startswith(removed)]
    additions = {
        "directml": ["torch-directml"],
        "onnx_directml": ["onnxruntime-directml"],
        "onnx_cpu": ["onnxruntime"],
        "rocm": ["# Install the matching ROCm PyTorch wheels using install-amd.sh/ps1 after selecting an official index URL."],
        "migraphx": ["# Install the matching ONNX Runtime MIGraphX wheel using install-amd.sh after selecting an official wheel URL."],
    }[target]
    output = bundle / "requirements.amd.txt"
    output.write_text("\n".join(preserved + [""] + additions) + "\n", encoding="utf-8")
    return output


def _command_display(command: Sequence[str], shell: str) -> str:
    if shell == "powershell":
        return " ".join('"{0}"'.format(item.replace('"', '\\"')) if " " in item else item for item in command)
    return " ".join("'{0}'".format(item.replace("'", "'\\''")) if re.search(r"\s", item) else item for item in command)


def render_report(report: Dict[str, object], target: str, commands: Sequence[Sequence[str]], prerequisites: Sequence[str], bundle_name: str = ".ai-portability") -> str:
    lines = [
        "# AMD AI Portability Report",
        "",
        "- Project: `{0}`".format(report["project"]),
        "- Platform: `{0}`".format(report["platform"]),
        "- Selected target: **{0}** — {1}".format(TARGET_INFO[target]["label"], TARGET_INFO[target]["summary"]),
        "- Scan: {0} files ({1} text files)".format(report["files_scanned"], report["text_files_scanned"]),
        "- Generated: {0}".format(report["scanned_at"]),
        "",
        "## Recommendation",
        "",
        str(next((candidate["reason"] for candidate in report["candidates"] if candidate["id"] == target), TARGET_INFO[target]["summary"])),
        "",
        "Official runtime guidance: {0}".format(TARGET_INFO[target]["docs"]),
        "",
        "## Findings",
        "",
    ]
    findings = report["findings"]
    if not findings:
        lines.append("No CUDA/NVIDIA lock-in signals were found in scanned text. Test the application before assuming its dependencies are portable.")
    else:
        for item in findings:
            paths = ", ".join("`{0}`".format(path) for path in item["paths"][:5]) or "no path recorded"
            lines.extend(["### [{0}] {1}".format(str(item["severity"]).upper(), item["title"]), "", str(item["detail"]), "", "Evidence: {0}".format(paths), ""])
    lines.extend(["## Generated bundle", "", "- `amd_portability.py`: explicit PyTorch/ONNX device and provider helper.", "- `requirements.amd.txt`: a non-destructive alternative dependency list when `requirements.txt` exists.", "- `install-amd.ps1` and `install-amd.sh`: reviewed installation commands."])
    if report["cuda_files"]:
        lines.append("- `hipified/`: HIP draft copies of CUDA sources. These require code review and a hipcc build; the original files are untouched.")
    lines.extend(["", "## Install plan", ""])
    if commands:
        for command in commands:
            lines.append("```sh\n{0}\n```".format(_command_display(command, "sh")))
    if prerequisites:
        for item in prerequisites:
            lines.append("- Prerequisite: {0}".format(item))
    lines.extend([
        "",
        "## Safety boundary",
        "",
        "`--apply` changes only simple Python device/provider literals and creates backups in `{0}/backups/`. It does not modify compiled binaries, delete CUDA packages, bypass license checks, or claim that a CUDA-only extension now works on AMD. Review the generated report and run the project test suite after installation.".format(bundle_name),
        "",
        "For Windows, DirectML is the broad fallback. For ONNX on current Linux ROCm, prefer MIGraphX; ROCm ONNX EP was deprecated upstream. Match every ROCm wheel to the official GPU/OS/Python compatibility matrix.",
    ])
    return "\n".join(lines) + "\n"


def _install_scripts(bundle: Path, commands: Sequence[Sequence[str]], prerequisites: Sequence[str]) -> None:
    ps = ["# Generated by portable_ai.py. Run inside the intended virtual environment.", "$ErrorActionPreference = 'Stop'", ""]
    sh = ["#!/usr/bin/env sh", "set -eu", "# Generated by portable_ai.py. Run inside the intended virtual environment.", ""]
    for command in commands:
        ps.append(_command_display(command, "powershell"))
        sh.append(_command_display(command, "sh"))
    if prerequisites:
        ps.extend(["", "# Complete these prerequisites before running a generated install command:"] + ["# " + item for item in prerequisites])
        sh.extend(["", "# Complete these prerequisites before running a generated install command:"] + ["# " + item for item in prerequisites])
    (bundle / "install-amd.ps1").write_text("\n".join(ps) + "\n", encoding="utf-8")
    (bundle / "install-amd.sh").write_text("\n".join(sh) + "\n", encoding="utf-8")


def prepare_project(
    project: str,
    target: str = "auto",
    platform_name: str = "auto",
    apply: bool = False,
    install: bool = False,
    force: bool = False,
    rocm_index_url: Optional[str] = None,
    migraphx_wheel: Optional[str] = None,
    python_executable: Optional[str] = None,
) -> Dict[str, object]:
    """Create an auditable portability bundle; opt-in to source edits/install."""
    report = scan_project(project, platform_name)
    selected = choose_target(report, target)
    root = Path(str(report["project"]))
    bundle = root / ".ai-portability"
    bundle.mkdir(exist_ok=True)
    python_executable = python_executable or sys.executable
    commands, prerequisites = installation_commands(selected, python_executable, rocm_index_url, migraphx_wheel)

    (bundle / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (bundle / "PORTABILITY_REPORT.md").write_text(render_report(report, selected, commands, prerequisites), encoding="utf-8")
    helper_path = root / "amd_portability.py"
    helper_backup = None
    if force and helper_path.exists():
        helper_backup = bundle / "backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") / "amd_portability.py"
        helper_backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(helper_path, helper_backup)
    helper_written = _write_if_changed(helper_path, RUNTIME_HELPER.format(target=selected), force=force)
    _alternative_requirements(root, bundle, selected)
    _install_scripts(bundle, commands, prerequisites)

    applied: List[Dict[str, object]] = []
    if apply:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_root = bundle / "backups" / timestamp
        for path in root.rglob("*.py"):
            if bundle in path.parents or path.name == "amd_portability.py" or any(part in SKIP_DIRS for part in path.parts):
                continue
            original = _read_text(path)
            if original is None:
                continue
            updated, changes = patch_python_source(original)
            if not changes or updated == original:
                continue
            relative = path.relative_to(root)
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            path.write_text(updated, encoding="utf-8")
            applied.append({"path": relative.as_posix(), "backup": backup.relative_to(root).as_posix(), "changes": changes})
        (bundle / "applied-changes.json").write_text(json.dumps(applied, indent=2) + "\n", encoding="utf-8")

    hipified: List[Dict[str, object]] = []
    if report["cuda_files"]:
        hip_root = bundle / "hipified"
        for relative in report["cuda_files"]:
            source_path = root / relative
            content = _read_text(source_path)
            if content is None:
                continue
            converted, replacements = convert_cuda_source(content)
            destination = hip_root / relative
            if destination.suffix.lower() == ".cu":
                destination = destination.with_suffix(".hip")
            elif destination.suffix.lower() == ".cuh":
                destination = destination.with_suffix(".hip.h")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(converted, encoding="utf-8")
            hipified.append({"source": relative, "draft": destination.relative_to(root).as_posix(), "replacements": replacements})
        (bundle / "hipified-manifest.json").write_text(json.dumps(hipified, indent=2) + "\n", encoding="utf-8")

    installations: List[Dict[str, object]] = []
    if install:
        if prerequisites:
            raise RuntimeError("Installation needs additional version-specific input:\n- " + "\n- ".join(prerequisites))
        for command in commands:
            completed = subprocess.run(command, cwd=str(root), text=True, capture_output=True)
            installation = {"command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
            installations.append(installation)
            if completed.returncode != 0:
                (bundle / "install-results.json").write_text(json.dumps(installations, indent=2) + "\n", encoding="utf-8")
                raise RuntimeError("Package installation failed. See {0}".format(bundle / "install-results.json"))
        (bundle / "install-results.json").write_text(json.dumps(installations, indent=2) + "\n", encoding="utf-8")

    return {
        "report": report, "target": selected, "bundle": str(bundle), "helper_written": helper_written,
        "helper_backup": str(helper_backup) if helper_backup else None, "applied": applied,
        "hipified": hipified, "installations": installations, "prerequisites": prerequisites,
    }


def _print_scan(report: Dict[str, object]) -> None:
    print("AMD AI Portability Kit {0}".format(VERSION))
    print("Project: {0}".format(report["project"]))
    print("Platform: {0}".format(report["platform"]))
    print("Scanned: {0} files; recommended target: {1}".format(report["files_scanned"], report["recommended_target"]))
    for candidate in report["candidates"]:
        print("  {0:>3}%  {1}: {2}".format(candidate["score"], candidate["id"], candidate["reason"]))
    if report["findings"]:
        print("Findings:")
        for item in report["findings"]:
            print("  [{0}] {1}".format(str(item["severity"]).upper(), item["title"]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit and prepare an AMD portability bundle for an AI project.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("scan", "Read-only accelerator dependency audit."), ("prepare", "Generate an AMD portability bundle.")):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("project", nargs="?", default=".", help="Project directory (default: current directory)")
        command.add_argument("--platform", default="auto", choices=("auto", "windows", "linux", "darwin"), help="Target operating system for planning.")
        command.add_argument("--json", action="store_true", help="Print the result as JSON.")
        if name == "prepare":
            command.add_argument("--target", default="auto", choices=("auto",) + tuple(sorted(TARGET_INFO)), help="Runtime target (default: best supported recommendation).")
            command.add_argument("--apply", action="store_true", help="Back up and apply simple Python device/provider literal changes.")
            command.add_argument("--install", action="store_true", help="Run only the generated package install command. Requires all target-specific inputs.")
            command.add_argument("--force", action="store_true", help="Replace an existing generated amd_portability.py helper.")
            command.add_argument("--python", default=None, help="Python executable/venv to install into (default: this Python).")
            command.add_argument("--rocm-index-url", default=None, help="Exact official ROCm PyTorch wheel index URL for --target rocm.")
            command.add_argument("--migraphx-wheel", default=None, help="Exact official ONNX Runtime MIGraphX wheel URL/path for --target migraphx.")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "scan":
            report = scan_project(arguments.project, arguments.platform)
            if arguments.json:
                print(json.dumps(report, indent=2))
            else:
                _print_scan(report)
            return 0
        result = prepare_project(
            arguments.project, arguments.target, arguments.platform, arguments.apply, arguments.install,
            arguments.force, arguments.rocm_index_url, arguments.migraphx_wheel, arguments.python,
        )
        if arguments.json:
            print(json.dumps(result, indent=2))
        else:
            _print_scan(result["report"])
            print("\nPrepared {0} bundle in: {1}".format(result["target"], result["bundle"]))
            if result["applied"]:
                print("Applied {0} reversible Python file change(s).".format(len(result["applied"])))
            if result["hipified"]:
                print("Created {0} HIP draft(s) for review.".format(len(result["hipified"])))
            if result["prerequisites"]:
                print("Installation was not run until these are supplied:")
                for item in result["prerequisites"]:
                    print("- " + item)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print("Error: {0}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
