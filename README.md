# 📦 Is Discretization Fusion All You Need for Collaborative Perception?

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

🌟 **News**: Our paper **"Is Discretization Fusion All You Need for Collaborative Perception?"** has been accepted to **ICRA 2025**!

---

## 📚 Table of Contents

- [📦 Is Discretization Fusion All You Need for Collaborative Perception?](#-is-discretization-fusion-all-you-need-for-collaborative-perception)
  - [📚 Table of Contents](#-table-of-contents)
  - [🧩 Code](#-code)
  - [💻 Installation](#-installation)
  - [🚀 Getting Started](#-getting-started)
    - [📥 Download Datasets](#-download-datasets)
    - [🏃‍♂️ Train](#️-train)
    - [🧪 Evaluation](#-evaluation)
  - [📖 Citation](#-citation)
  - [🙏 Acknowledgements](#-acknowledgements)

---

## 🧩 Code

Code is available in this repository. Please refer to individual scripts and folders for usage details.

---

## 💻 Installation

Please refer to the detailed documentation in [`docs/INSTALL.md`](./docs/INSTALL.md).

You can also install required packages via pip:

```bash
pip install -r requirements.txt
```

Or create a conda environment if `environment.yml` is provided:

```bash
conda env create -f environment.yml
conda activate ACCO
```

---

## 🚀 Getting Started

### 📥 Download Datasets

- **OPV2V**: https://mobility-lab.seas.ucla.edu/opv2v/
- **DAIR-V2X**: https://github.com/AIR-THU/DAIR-V2X?tab=readme-ov-file#dataset

### 🏃‍♂️ Train

To start training, simply run:

```bash
./train.sh
```

### 🧪 Evaluation

To run evaluation (inference):

```bash
./inference.sh
```


---

## 📖 Citation

If you use this work, please consider citing our paper.

```bibtex
@article{yang2025discretization,
  title={Is Discretization Fusion All You Need for Collaborative Perception?},
  author={Yang, Kang and Bu, Tianci and Li, Lantao and Li, Chunxu and Wang, Yongcai and Li, Deying},
  journal={arXiv preprint arXiv:2503.13946},
  year={2025}
}
```
---

## 🙏 Acknowledgements

- Thanks to the authors of [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) and [Where2comm](https://github.com/MediaBrain-SJTU/where2comm) for providing excellent codebases.
- We also appreciate the datasets provided by [DAIR-V2X](https://thudair.baai.ac.cn/index) and [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/).

---
