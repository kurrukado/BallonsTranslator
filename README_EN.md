> [!IMPORTANT]  
> **If you share machine-translated results publicly without thorough translation or proofreading by experienced translators, please clearly indicate that it is a machine translation.**  
> **This tool was developed entirely with AI, so minor quirks may exist and some folder structures might look unconventional.**

# 🎈 BalloonsTranslator — Low-Cost & Local AI Edition

[![AI Architecture](https://img.shields.io/badge/Architecture-Low--Cost%20%26%20Local%20AI-brightgreen.svg)](#)
[![LLM Engine](https://img.shields.io/badge/LLM-Gemini%20Flash%20Series-blue.svg)](#)
[![Local Inpainting](https://img.shields.io/badge/Inpaint-LaMa%20Large%20(Local%20CUDA)-orange.svg)](#)
[![Text Detection](https://img.shields.io/badge/Detector-ComicTextDetector%202K-purple.svg)](#)
[![License](https://img.shields.io/badge/License-GPL--3.0-yellow.svg)](LICENSE)

[English](README_EN.md) | [Tiếng Việt](README.md) | [简体中文](doc/README_CN.md) | [日本語](doc/README_JA.md)

---

### 🌟 About This Fork

This fork aims to minimize translation costs and enhance output quality by pairing low-cost cloud LLMs with dedicated local neural AI:

- **Context-Aware Translation**: Powered by **Google Gemini Flash & Flash-Lite** with chapter-batch aggregation, authentic scanlation pronoun inference based on character dynamics, and strict anti-hallucination schema locking.
- **On-Device Background Inpainting & Erasing**: Powered by **Clean-Canvas LaMa Large** combined with **Laplacian Multi-Band Pyramid Blending** running locally on your GPU/CPU.
- **Local Automated Text Detection**: Utilizes **ComicTextDetector 2K High-DPI (1536px)** with enhanced boundary sensitivity to accurately isolate complex speech bubbles and free-floating text outside dialogue frames.
- **Local Multi-Tier OCR**: Integrates native **Windows Media OCR** (built into Windows) coupled with **PaddleOCR**, cleanly extracting comic and manga typography.
- **Specialized Scanlation Typography**: Automatically classifies dialogue emotion to apply tailored manga/comic font suites (`Yuki-CCMarianChurchlandJournal` for regular speech and `CCWildWordsRoman` for shouts/action screams).

---

### 🙏 Credits & Upstream Attribution

- **License**: This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](LICENSE), fully preserving all original copyright notices and showing utmost respect to upstream creators.
- **Fonts**: Shared by the manga and comic scanlation community.

All foundational GUI architecture, canvas rendering systems, and core algorithms are credited to [@dmMaze](https://github.com/dmMaze) and upstream contributors. Huge thanks to the original authors!

---

<p align="center">
  <img src="doc/src/ui0.jpg" alt="BalloonsTranslator User Interface">
</p>
<p align="center">
  <em>BalloonsTranslator Interface (Low-Cost & Local AI Edition)</em>
</p>

# Installation Guide

## System Requirements
- **Operating System**: Windows 10/11 (64-bit).
- **Python**: Recommended Python 3.10 to 3.12 (or Python 3.14 via frozen mode).
- **GPU (Graphics Card)**: NVIDIA GPU with CUDA support recommended for accelerated inpainting and detection; CPU execution is fully supported.

## Running from Source (Fastest & Recommended)

1. **Install Python & Git**:  
   Download and install [Python](https://www.python.org/downloads/) (ensure "Add Python to PATH" is checked) and [Git](https://git-scm.com/downloads).

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/kurrukado/BallonsTranslator.git
   cd BallonsTranslator
   ```

3. **Launch the Application**:
   - **Method 1 (Recommended)**: Double-click `start.bat` to automatically launch the proxy server and the translation interface.
   - **Method 2 (Command Line)**:
     ```bash
     python launch.py --frozen
     ```

---

# Usage Guide

## 1-Click Folder Translation
1. Open the application, click on the **Settings (gear icon)**:
   - Configure your Gemini API Key in the settings dialog (or in `config/secrets.json`).
   - Select translator: `gemini` (or `gemini-proxy`).
   - Select source language (Source Language): e.g., `Japanese` or `English`.
   - Select target language (Target Language): e.g., `Vietnamese`.
2. Click the **Folder** icon to open a folder containing your manga or comic images.
3. Click the **Run** button and wait for the pipeline to complete text detection, inpainting, and translation automatically.

<p align="center">
  <img src="doc/src/run.gif" alt="Automated 1-Click Translation">
</p>

---

## Interactive Manual Editing Tools

### Inpaint Healing Brush
<p align="center">
  <img src="doc/src/imgedit_inpaint.gif" alt="Inpaint Healing Brush">
</p>

### Rectangle Tool
<p align="center">
  <img src="doc/src/rect_tool.gif" alt="Rectangle Tool">
</p>

- Hold **Left Mouse Button** and drag a rectangle to erase text inside the box.
- Hold **Right Mouse Button** and drag a rectangle to undo inpainting and restore the original image.
- Check **Auto** to inpaint immediately upon releasing the mouse, or press `Space` / the `Inpaint` button. Press `Ctrl+D` to deselect.

### Text Editing & Auto-Layout
<p align="center">
  <img src="doc/src/textedit.gif" alt="Text Editing">
</p>

<p align="center">
  <img src="doc/src/multisel_autolayout.gif" alt="Batch Text Layout">
</p>

---

# Handy Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Ctrl + Z` / `Ctrl + Y` | Undo / Redo |
| `A` / `D` or `PageUp` / `PageDown` | Previous / Next page (auto-saves current page) |
| `T` | Switch to Text Edit Mode |
| `W` | Activate new text block creation mode (then drag right mouse on canvas) |
| `P` | Switch to Inpaint Brush Mode |
| `Ctrl + (+ / -)` or `Mouse Wheel` | Zoom in / Zoom out canvas |
| `Ctrl + A` | Select all text blocks on current page |
| `Ctrl + F` | Find text in current page |
| `Ctrl + G` | Global find across all pages |
| `Ctrl + B` / `Ctrl + I` / `Ctrl + U` | Bold / Italic / Underline selected text |
| `Alt + Arrow Keys` or `Alt + WASD` | Navigate between text blocks |
| `0 - 9` | Adjust opacity of translated layer / original image |

---

# Headless CLI Mode

You can run batch translations from the command line without opening the GUI:
```bash
python launch.py --headless --exec_dirs "D:/Manga/Chapter_01,D:/Manga/Chapter_02"
```
All settings (detection model, source/target languages) are loaded automatically from `config/config.json`.

---

# License & Disclaimer

- Released under the [GNU General Public License v3.0](LICENSE).
- This tool is developed for educational, AI research, and scanlation community assistance. Please respect the copyright of original comic creators.
