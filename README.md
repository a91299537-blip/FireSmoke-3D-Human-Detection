# Geometry-Confidence-Coupled Real-Time 3D Human Detection

Core implementation accompanying the manuscript **"Geometry-Confidence-Coupled Real-Time 3D Human Detection from Point Clouds in Fire-Smoke Environments"**.

This repository contains the method-specific source files, STCrowd data adaptation code, and experiment configurations. It is distributed as an overlay for [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) to keep the release small and to make the upstream dependency explicit.

## Release Scope

- Sparse candidate assignment, sparse box decoding, and associated regression losses.
- Sparse feature diffusion backbone and confidence-aware training support.
- Ordered state-space backbone and variance-aligned low-bit quantization path.
- Fire-smoke point-cloud degradation used during training.
- STCrowd conversion, evaluation, and experiment configurations.

The internal implementation names predate the final manuscript terminology. The mapping is:

| Manuscript component | Main implementation files |
| --- | --- |
| SGCE | `sparse_dynamic_head.py`, `centernet_utils.py`, `loss_utils.py` |
| SC-DCM | `fshnet_base.py`, smoke-aware augmentation code |
| OTC-SSA and low-bit path | `gl_ssm.py`, `gl_ssm_qat4bit.yaml` |

## Upstream Version

The overlay targets OpenPCDet commit:

```text
233f849829b6ac19afb8af8837a0246890908755
```

## Installation

1. Clone and pin OpenPCDet.

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
git checkout 233f849829b6ac19afb8af8837a0246890908755
```

2. Install OpenPCDet following its official instructions, including the CUDA-compatible PyTorch and `spconv` builds.

3. Install the additional dependencies used by the released modules.

```bash
pip install -r requirements-extra.txt
```

4. Apply this repository's overlay to the clean OpenPCDet checkout.

```bash
python scripts/install_overlay.py --openpcdet-root /path/to/OpenPCDet --force
```

The installer intentionally requires `--force` because it replaces upstream files. Apply it only to a clean checkout at the pinned commit.

## Data Preparation

Download STCrowd from the [official project page](https://4dvlab.github.io/STCrowd/index.html). The dataset is not redistributed by this repository.

Convert the official files into the KITTI-compatible layout expected by the configuration:

```bash
python scripts/convert_stcrowd.py \
  --stcrowd-root /path/to/STCrowd_official \
  --output-dir /path/to/OpenPCDet/data/stcrowd
```

Then generate OpenPCDet information files and the ground-truth database:

```bash
cd /path/to/OpenPCDet
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos \
  tools/cfgs/dataset_configs/stcrowd_dataset.yaml
```

## Training and Evaluation

From the OpenPCDet `tools` directory:

```bash
python train.py --cfg_file cfgs/kitti_models/fshnet_stcrowd.yaml
python test.py --cfg_file cfgs/kitti_models/fshnet_stcrowd.yaml \
  --ckpt /path/to/checkpoint.pth
```

The ordered state-space and 4-bit configurations are provided as research configurations in `openpcdet_overlay/tools/cfgs/kitti_models/`.

## Reproducibility Status

This initial release contains the auditable core implementation and configurations. Raw datasets, experiment logs, manuscript files, and local visualization artifacts are deliberately excluded. Model weights are not labeled as the manuscript's final weights until their checksum and reported metrics have been independently matched to the final manuscript table.

## License and Attribution

This work is released under the Apache License 2.0 and includes code derived from OpenPCDet. See [LICENSE](LICENSE) and [NOTICE](NOTICE). STCrowd remains subject to its original license and terms.

## Citation

Please cite the associated paper after its bibliographic record is available. A machine-readable placeholder is provided in [CITATION.cff](CITATION.cff).

