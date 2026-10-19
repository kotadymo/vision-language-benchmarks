# Development and benchmark environment

## Verified development host

`vlbench doctor --device cuda` was run in the development environment. It reported:

| Component | Observed value |
| --- | --- |
| Operating system | Windows (`win32`; Windows 11 development host) |
| Python | 3.12.10 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA runtime | 13.0 |
| CUDA available | Yes |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| GPU memory | 16 GiB class (17,170,956,288 bytes reported) |
| Compute capability | 8.9 |
| NVIDIA driver | 610.88 |
| BF16 support | Yes |

This is an environment diagnostic, not a benchmark result. No model was loaded and no
evaluation performance was measured. The doctor command reports only whether Hugging Face
authentication is present; it never prints or records a token. Storage availability changes
over time and is intentionally not published here.

## Pinned software

Python dependencies and the lockfile are defined by [`pyproject.toml`](../pyproject.toml)
and [`uv.lock`](../uv.lock). The project targets Python 3.12. Optional model-family extras
keep normal unit tests independent of PyTorch, GPU libraries, and model packages. Install
only the extras needed for a planned run; see [reproducibility](reproducibility.md).

## CI

GitHub Actions checks Python 3.12 on Windows and Linux. It validates the lockfile, installs
the base project plus development tools, runs Ruff and pytest, and checks whitespace. The
suite uses synthetic inputs and mocks; CI does not download benchmark datasets or model
weights, require CUDA, or claim real-model accuracy.
