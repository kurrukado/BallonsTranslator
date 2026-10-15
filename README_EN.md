> [!IMPORTANT]  
> **If used, please clearly credit that this is machine translation. This tool was developed entirely by AI so there may be flaws; this fork strongly focuses on translating into Vietnamese.**

# BalloonsTranslator — Low-Cost & Local AI Edition
[![License](https://img.shields.io/badge/License-GPL--3.0-yellow.svg)](LICENSE)

[English](README_EN.md) | [Tiếng Việt](README.md) | [简体中文](doc/README_CN.md) | [日本語](doc/README_JA.md)

---

### About This Fork

This fork aims to optimize costs and enhance translation quality using low-cost and local AI models:

- **Context-Aware Translation**: Powered by **Google Gemini Flash & Flash-Lite** with chapter-batch aggregation, natural scanlation pronoun inference based on character roles/relationships, and strict anti-hallucination schema locking.
- **On-Device Inpainting & Erasing**: Powered by **Clean-Canvas LaMa Large** combined with **Laplacian Multi-Band Pyramid Blending** running directly on GPU/CPU.
- **Local Automated Text Detection**: Utilizes **ComicTextDetector 2K High-DPI (1536px)** with enhanced boundary sensitivity to accurately isolate complex speech bubbles and free-floating text outside dialogue frames.
- **Local Multi-Tier OCR**: Integrates native **Windows Media OCR** (built into Windows) coupled with **PaddleOCR**.
- **Built-in Fonts**: Yuki-CCMarianChurchlandJournal and CCWildWordsRoman (sourced from a Facebook group).

---

### 🙏 Credits & Upstream Attribution

All foundational contributions to GUI architecture, canvas rendering, and core algorithms belong to [@dmMaze](https://github.com/dmMaze) and original upstream contributors. Sincere thanks to the original authors!

---

# Installation Guide

## System Requirements
- **Operating System**: Windows 10/11 (64-bit).
- **Python**: Recommended Python 3.10 to 3.12 (or Python 3.14 via frozen mode).
- **GPU (Graphics Card)**: NVIDIA GPU (CUDA support) recommended for faster inpainting and detection; CPU execution is fully supported.

## Running from Source (Fastest & Recommended)

1. **Install Python & Git**:  
   Download and install [Python](https://www.python.org/downloads/) (check "Add to PATH") and [Git](https://git-scm.com/downloads).

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/kurrukado/BallonsTranslator.git
   cd BallonsTranslator
   ```

3. **Launch the Application**:
   - **Method 1 (Recommended)**: Double-click `start.bat` to automatically launch the proxy and translation interface.
   - **Method 2 (Command Line)**:
     ```bash
     python launch.py --frozen
     ```

---

# License & Disclaimer

- This project is released under the [GNU General Public License v3.0](LICENSE).
- This tool is developed for educational and AI research purposes, and to support the scanlation community. Please respect the copyright of the original comic creators.
