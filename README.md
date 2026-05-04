# Neural Vertex Features

Code release for "Vertex Features for Neural Global Illumination".

## Requirements

**Platform**
- Windows 10 / Ubuntu 20.04
- CUDA 12.2

**Dependencies**
- Python 3.10
- PyTorch 2.1.0
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
- numpy, tqdm, torch_scatter
- imgui, glfw, pyopengl, pycuda (for interactive viewer)

## Compiling Mitsuba 3 from Source

This project uses a [modified fork](https://github.com/woAIxuexiSR/mitsuba3) of Mitsuba 3 (included as a submodule) that exposes barycentric coordinates from ray intersections. It also requires the `cuda_rgb` variant, which is not available in pre-built binaries. You need to compile it from source.

Refer to the [official documentation](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html) for building on different platforms. On Windows:

```bash
cd mitsuba
cmake -G "Visual Studio 17 2022" -A x64 -B build
```

Enable the `cuda_rgb` variant in `build/mitsuba.conf` (~line 86), then build:

```bash
cmake --build build --config Release
```

After building, run the `setpath` script to configure environment variables (`PATH`, `PYTHONPATH`):

```bash
# Windows PowerShell
cd mitsuba\build\Release
.\setpath.ps1

# Linux
source mitsuba/build/setpath.sh
```

## Usage

Scene assets are stored under `scenes/<scene-name>/`. Pretrained weights and config files are stored under `pretrained_model/<scene-name>/`.

### Static Scenes

**Train:**

```bash
python render_static/train.py -c pretrained_model/living-room/static.json -m pretrained_model/living-room/static.pth
```

**Render:**

```bash
python render_static/render_img.py [-t type] [-s spp] [-c pretrained_model/living-room/static.json] [-m pretrained_model/living-room/static.pth] [-o output.exr]
```

- `-t`: rendering method: `LHS`, `RHS`, `path`, or `level`
- `-s`: samples per pixel
- `-o`: output file path

### Dynamic Scenes

**Train:**

```bash
python render_dynamic/train.py -c pretrained_model/dining-room/dynamic.json -m pretrained_model/dining-room/dynamic.pth
```

**Render:**

```bash
python render_dynamic/render_img.py [-t type] [-s spp] [-c pretrained_model/dining-room/dynamic.json] [-m pretrained_model/dining-room/dynamic.pth] [-o output.exr]
```

- `-t`: rendering method: `LHS`, `RHS`, or `path`
- `-s`: samples per pixel
- `-o`: output file path

### Metrics

```bash
python utils/metrics.py <image.exr> <reference.exr> <metric>
```

- `metric`: `MSE`, `relMSE`, `MAPE`, `MAE`, or `SMAPE`

### Interactive Viewer

```bash
python test.py -c pretrained_model/dining-room/dynamic.json -m pretrained_model/dining-room/dynamic.pth
```

## Config File

| Field | Description |
|---|---|
| `scene` | Path to the Mitsuba scene file (`.xml`) |
| `output` | Output directory name (saved under `result/`) |
| `animation` | Path to the animation file (`.json`, dynamic only) |
| `v` | Animation variable values (dynamic only) |
| `cam_v` | Camera animation value used by the dynamic render/viewer entrypoints |

**Training parameters** (`train`):

| Field | Description |
|---|---|
| `use_subdivide` | Enable adaptive mesh subdivision |
| `use_adaptive_rhs` | Progressively increase RHS samples during training |
| `loss` | Loss function (e.g., `weighted_normed_semi_l2`) |
| `rhs_samples` | Number of samples for the right-hand side |
| `batch_size` | Batch size |
| `steps` | Number of training steps |
| `learning_rate` | Learning rate |
| `save_interval` | Model checkpoint interval (in steps) |
| `model_name` | Checkpoint filename |

**Model parameters** (`model`):

| Field | Description |
|---|---|
| `type` | Model type (`VModel` for static, `DModel` for dynamic) |
| `feature_dim` | Per-vertex feature dimension |
| `n_hidden_layers` | Number of hidden layers |
| `n_neurons` | Neurons per hidden layer |
| ... | Additional dynamic-model parameters such as `grid_type_2d`, `n_levels_2d`, `base_resolution_2d`, `per_level_scale_2d`, and `vvmlp_output_dims` |

Example configs:
- Static: [pretrained_model/living-room/static.json](pretrained_model/living-room/static.json)
- Dynamic: [pretrained_model/dining-room/dynamic.json](pretrained_model/dining-room/dynamic.json)
