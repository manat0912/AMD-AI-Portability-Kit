# AMD AI Portability Kit

An auditable migration assistant for AI projects that need to run without an
NVIDIA CUDA dependency. It retains the existing CUDA-to-HIP converter and adds
a standalone project scanner and runtime planner for ROCm/HIP, DirectML, ONNX
Runtime DirectML, MIGraphX, and CPU fallback.

## Quick start: use on any AI project

Copy [`portable_ai.py`](portable_ai.py) into the target project's root, then:

```powershell
python portable_ai.py scan .
python portable_ai.py prepare . --target auto
```

`prepare` creates an `.ai-portability/` report, installation scripts,
alternative requirements file, and an explicit `amd_portability.py` runtime
helper. Nothing in the target application is overwritten by default.

After reviewing the report, use `--apply` to make only simple, backed-up Python
device/provider substitutions, and `--install` to install a selected runtime:

```powershell
# Windows PyTorch application (DirectML)
python portable_ai.py prepare . --target directml --apply --install

# Windows ONNX application
python portable_ai.py prepare . --target onnx_directml --apply --install

# Linux ROCm application: supply a wheel index that exactly matches your
# supported ROCm, Python and GPU combination.
python portable_ai.py prepare . --target rocm --rocm-index-url https://YOUR-OFFICIAL-ROCM-WHEEL-INDEX --apply --install
```

The script deliberately does not claim it can convert precompiled CUDA wheels,
TensorRT engines, proprietary NVIDIA binaries, or custom CUDA/Triton kernels.
It identifies those blockers and produces HIP draft copies only when source is
available. These need an AMD/upstream replacement or a reviewed HIP rebuild.

Current guidance is encoded as safe defaults: DirectML is a Windows fallback;
ROCm is the main PyTorch/HIP route; and ONNX on modern Linux ROCm should prefer
MIGraphX. Always check the current GPU/OS/Python compatibility matrix before
installing ROCm wheels.

## Files
- `portable_ai.py`: Self-contained scanner, planner, source adapter, and optional installer to copy into other AI projects
- `pinokio.js`: Main Pinokio menu configuration
- `install.js`: Installation script for Python environment and dependencies
- `start.js`: Launch script starting the Gradio UI daemon
- `update.js`: Script for updating the app
- `reset.js`: Environment reset script
- `icon.svg`: Application logo
- `app/`: Python backend and Gradio web interface
