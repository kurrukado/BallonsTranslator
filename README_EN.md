> [!IMPORTANT]  
> **If you share machine-translated results publicly without thorough translation or proofreading by experienced translators, please clearly indicate that it is a machine translation.**

# 🎈 BalloonsTranslator — 100% Free AI Models Edition

> **A specialized, production-ready fork of BalloonsTranslator optimized for 100% Free & Local AI Pipelines (Google Gemini API Free Tier, On-Device LaMa Inpainting, ComicTextDetector 2K, Windows OCR & Vietnamese Scanlation Typography).**

[![Free AI Models](https://img.shields.io/badge/AI%20Cost-100%25%20Free-brightgreen.svg)](#)
[![Google Gemini API](https://img.shields.io/badge/LLM-Gemini%20Flash%20Free%20Tier-blue.svg)](#)
[![Local Inpainting](https://img.shields.io/badge/Inpaint-LaMa%20Large%20(Local%20CUDA)-orange.svg)](#)
[![Text Detection](https://img.shields.io/badge/Detector-ComicTextDetector%202K-purple.svg)](#)
[![License](https://img.shields.io/badge/License-GPL--3.0-yellow.svg)](LICENSE)

[English](README_EN.md) | [Tiếng Việt](README.md) | [简体中文](doc/README_CN.md) | [日本語](doc/README_JA.md)

---

### 🌟 Why This Free AI Edition?

This fork was created to eliminate all costly API subscriptions (such as OpenAI GPT-4 or DeepL Pro). It replaces the entire translation, inpainting, and text detection pipeline with **the highest quality 100% Free and Local AI technologies available today**:

| Pipeline Component | Original / Other Forks | Free AI Edition (This Fork) | Cost |
| :--- | :--- | :--- | :---: |
| **Smart Translation (LLM)** | Dependent on paid DeepL / OpenAI; prone to rate limits & disjointed phrasing | **Google Gemini Flash & Lite** via built-in `gemini-proxy` with automatic load balancing. Multi-bubble sentence stitching, contextual manga pronouns, and 100% anti-hallucination. | **$0 (Free Tier)** |
| **Text Removal (Inpaint)** | Gradient stepping artifacts, visible seams, pixel leakage | **Clean-Canvas LaMa Large** combined with **Laplacian Multi-Band Pyramid Blending**. Seamlessly restores sloped screentones, preserving 100% of original unmasked pixels (`Unmasked MAE = 0.000000`). | **$0 (Local GPU)** |
| **Text Detection** | Misses free-floating text, downscales 2K high-res panels | **ComicTextDetector 2K High-DPI (1536px)** with high-sensitivity thresholds (`text_thresh=0.25, link_thresh=0.20`), capturing 100% of open bubbles and narration boxes. | **$0 (Local GPU)** |
| **OCR (Text Recognition)** | Requires paid 3rd-party APIs or heavy setups | **Windows Media OCR** built into Windows + **PaddleOCR** offline fallback. | **$0 (Offline)** |
| **Typography & Fonts** | Single generic font for all dialogue | **Automatic 2-Font Lettering System**: Normal dialogue (`Yuki-CCMarianChurchlandJournal`) and Shout/Action (`CCWildWordsRoman`) with full Vietnamese diacritics. | **$0 (Built-in)** |

---

### 🙏 Credits & Upstream Attribution

This project is an enhanced fork of the remarkable open-source [**BallonsTranslator**](https://github.com/dmMaze/BallonsTranslator) created by [**@dmMaze**](https://github.com/dmMaze).

- **Original Author / Upstream Repository**: [dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator)
- **License**: Released under the terms of the [GNU General Public License v3.0 (GPL-3.0)](LICENSE), strictly preserving all original copyright notices.
- **Fork Objectives**: Maintained by [@kurrukado](https://github.com/kurrukado/BallonsTranslator) with a dedicated focus on **100% Free & Local AI Pipelines** (Google Gemini Free-Tier multi-key load balancing, Clean-Canvas LaMa inpainting, ComicTextDetector 2K High-DPI, and Vietnamese scanlation typography presets) to enable high-quality manga translation with zero ongoing API costs.

All foundational GUI architecture, canvas rendering systems, and core algorithms are credited to [@dmMaze](https://github.com/dmMaze) and upstream contributors. Huge thanks to them for creating this project!

---

<p align="center">
  <img src="doc/src/ui0.jpg" alt="BalloonsTranslator Free AI Edition UI">
</p>
<p align="center">
  <em>BalloonsTranslator UI (Free AI Models Edition)</em>
</p>

---

# Features

* **Fully Automated 1-Click Translation**  
  - Automatic comic text detection, recognition, inpainting, and translation.
  - Typesetting estimation matches the original text format (color, outline, angle, alignment, and size).
  - Works seamlessly with Japanese Manga, Korean Manhwa, and Western Comics.
  - Integrated `gemini-proxy` automatically rotates multiple free Gemini API keys, bypassing rate limits effortlessly.

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

3. **Configure Free Gemini API Keys**:
   - Get 1 or more free API keys from [Google AI Studio](https://aistudio.google.com/).
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
