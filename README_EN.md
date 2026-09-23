> [!IMPORTANT]  
> **If you share machine-translated results publicly without thorough translation or proofreading by experienced translators, please clearly indicate that it is a machine translation.**

# 🎈 BalloonsTranslator — Low-Cost & Local AI Edition

> **An enhanced fork of BalloonsTranslator optimized for cost-effective manga/comic translation using low-cost LLMs combined with high-performance local AI engines.**

[![AI Architecture](https://img.shields.io/badge/Architecture-Low--Cost%20%26%20Local%20AI-brightgreen.svg)](#)
[![LLM Engine](https://img.shields.io/badge/LLM-Gemini%20Flash%20Series-blue.svg)](#)
[![Local Inpainting](https://img.shields.io/badge/Inpaint-LaMa%20Large%20(Local%20CUDA)-orange.svg)](#)
[![Text Detection](https://img.shields.io/badge/Detector-ComicTextDetector%202K-purple.svg)](#)
[![License](https://img.shields.io/badge/License-GPL--3.0-yellow.svg)](LICENSE)

[English](README_EN.md) | [Tiếng Việt](README.md) | [简体中文](doc/README_CN.md) | [日本語](doc/README_JA.md)

---

### 🌟 About This Fork

This fork is designed to provide cost-effective comic and manga translation workflows by uniting high-throughput, low-cost large language models (LLMs) with specialized, on-device local AI models:

- **Context-Aware Translation (Low-Cost LLM)**: Utilizes the **Google Gemini Flash & Flash-Lite** family managed by a built-in `gemini-proxy` with multi-key load balancing. Features chapter-batch aggregation for multi-bubble sentence stitching, contextual pronoun alignment, and anti-hallucination validation.
- **On-Device Inpainting (Local Neural AI)**: Powered by **Clean-Canvas LaMa Large** and **Laplacian Multi-Band Pyramid Blending** executing locally on your GPU/CPU, seamlessly reconstructing halftones and screen textures while preserving original unmasked artwork (`Unmasked MAE = 0.000000`).
- **High-DPI Text Detection (Local AI)**: Employs **ComicTextDetector 2K High-DPI (1536px)** with enhanced boundary sensitivity to detect standard bubbles, open frames, and free-floating text.
- **Multi-Tier Local OCR**: Combines native **Windows Media OCR** with **PaddleOCR** offline fallback for fast and accurate character recognition without cloud dependencies.
- **Automated Typography & Scanlation Presets**: Distinguishes spoken dialogue from shouts/SFX and applies tailored scanlation font styles automatically.

---

### 🙏 Credits & Upstream Attribution

This project is an enhanced fork of the remarkable open-source [**BallonsTranslator**](https://github.com/dmMaze/BallonsTranslator) created by [**@dmMaze**](https://github.com/dmMaze).

- **Original Author / Upstream Repository**: [dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator)
- **License**: Released under the terms of the [GNU General Public License v3.0 (GPL-3.0)](LICENSE), strictly preserving all original copyright notices.
- **Fork Objectives**: Maintained by [@kurrukado](https://github.com/kurrukado/BallonsTranslator) with a dedicated focus on combining low-cost LLM translation with on-device local AI processing to deliver efficient, high-quality manga and comic localization.

All foundational GUI architecture, canvas rendering systems, and core algorithms are credited to [@dmMaze](https://github.com/dmMaze) and upstream contributors. Huge thanks to them for creating this project!

---

<p align="center">
  <img src="doc/src/ui0.jpg" alt="BalloonsTranslator UI">
</p>
<p align="center">
  <em>BalloonsTranslator UI (Low-Cost & Local AI Edition)</em>
</p>

---

# Features

* **Fully Automated 1-Click Translation**  
  - Automatic comic text detection, recognition, inpainting, and translation.
  - Typesetting estimation matches the original text format (color, outline, angle, alignment, and size).
  - Works seamlessly with Japanese Manga, Korean Manhwa, and Western Comics.
  - Integrated `gemini-proxy` automatically rotates multiple Gemini API keys, ensuring steady throughput and seamless model fallback.

* **Image Editing & Inpainting**  
  - Supports interactive mask editing and inpaint healing brush (similar to Photoshop's Spot Healing Brush).
  - Handles extreme aspect ratio webtoons and ultra-high-resolution 2K/4K pages.

* **Text Editing & Typesetting**  
  - WYSIWYG rich text editor with full styling controls.
  - Supports [Text Style Presets](https://github.com/dmMaze/BallonsTranslator/pull/311) for quick styling application.
  - Page-wide and project-wide search & replace.
  - Import/Export translated dialogue to Microsoft Word (.docx) documents for easy proofreading.

---

# Installation & Getting Started

## System Requirements
- **OS**: Windows 10/11 (64-bit).
- **Python**: Recommended Python 3.10 to 3.12 (or Python 3.14 via frozen mode).
- **GPU**: NVIDIA GPU with CUDA support recommended for fast inpainting and detection; CPU execution is supported as fallback.

## Running from Source (Recommended)

1. **Install Python & Git**:  
   Download and install [Python](https://www.python.org/downloads/) (check "Add Python to PATH") and [Git](https://git-scm.com/downloads).

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/kurrukado/BallonsTranslator.git
   cd BallonsTranslator
   ```

3. **Configure Gemini API Keys**:
   - Get 1 or more API keys from [Google AI Studio](https://aistudio.google.com/).
   - Open `gemini-proxy/api-key.txt` (or copy from `api-key.txt.example`) and paste your keys, one per line:
     ```text
     AIzaSyYourFirstGeminiApiKey...
     AIzaSyYourSecondGeminiApiKey...
     ```

4. **Launch the Application**:
   - **Method 1 (Recommended)**: Double-click `start.bat` to launch both the proxy server and the GUI automatically.
   - **Method 2 (Command Line)**:
     ```bash
     python launch.py --frozen
     ```

---

# Usage Guide

## 1-Click Folder Translation
1. Open the application and click the **Settings** icon:
   - Select translator: `gemini` (or `gemini-proxy`).
   - Set source language: e.g. `Japanese` or `English`.
   - Set target language: `Vietnamese` or your preferred language.
2. Click the **Folder** icon to open a folder containing manga/comic image files.
3. Click the **Run** button and wait for the automated pipeline to complete.

<p align="center">
  <img src="doc/src/run.gif" alt="1-Click Translation">
</p>

---

## Interactive Editing Tools

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
- Check **Auto** to inpaint immediately upon releasing the mouse, or press `Space` / `Inpaint` button. Press `Ctrl+D` to deselect.

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
| `W` | Activate new text block creation (then drag right mouse on canvas) |
| `P` | Switch to Inpaint Brush Mode |
| `Ctrl + (+ / -)` or `Mouse Wheel` | Zoom in / Zoom out canvas |
| `Ctrl + A` | Select all text blocks on the current page |
| `Ctrl + F` | Find in current page |
| `Ctrl + G` | Global find across all pages |
| `Ctrl + B` / `Ctrl + I` / `Ctrl + U` | Bold / Italic / Underline selected text |
| `Alt + Arrow Keys` or `Alt + WASD` | Navigate between text blocks |
| `0 - 9` | Adjust transparency of translation layer / original image |

---

# Headless CLI Mode

Run batch translations from the command line without opening the GUI:
```bash
python launch.py --headless --exec_dirs "D:/Manga/Chapter_01,D:/Manga/Chapter_02"
```
All settings (detection model, source/target languages) are loaded from `config/config.json`.

---

# License

- Released under the [GNU General Public License v3.0](LICENSE).
- This tool is developed for educational, AI research, and scanlation community assistance. Please respect the copyright of original comic creators.
