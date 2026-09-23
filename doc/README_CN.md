> [!IMPORTANT]  
> **如打算公开分享本工具的机翻结果，且没有有经验的译者进行过完整的翻译或校对，请在显眼位置注明机翻。**

# 🎈 BalloonsTranslator — 100% 免费 AI 模型版 (Free AI Models Edition)

> **专为 100% 免费与本地 AI 流程（Zero-Cost / Free-Tier AI Pipelines）深度优化的 BalloonsTranslator 分支版本。集成 Google Gemini API 免费层多密钥负载均衡、Clean-Canvas LaMa 本地修复、ComicTextDetector 2K 高清文本检测及 Vietnamese/多语言汉化排版预设。**

[![Free AI Models](https://img.shields.io/badge/AI%20Cost-100%25%20Free-brightgreen.svg)](#)
[![Google Gemini API](https://img.shields.io/badge/LLM-Gemini%20Flash%20Free%20Tier-blue.svg)](#)
[![Local Inpainting](https://img.shields.io/badge/Inpaint-LaMa%20Large%20(Local%20CUDA)-orange.svg)](#)
[![Text Detection](https://img.shields.io/badge/Detector-ComicTextDetector%202K-purple.svg)](#)
[![License](https://img.shields.io/badge/License-GPL--3.0-yellow.svg)](../LICENSE)

[简体中文](README_CN.md) | [English](../README_EN.md) | [Tiếng Việt](../README.md) | [日本語](README_JA.md)

---

### 🌟 为什么选择本分支版本？ (Why This Fork?)

本分支旨在**彻底免除翻译者与汉化组的高昂 API 订阅成本**（如 OpenAI GPT-4 或 DeepL Pro）。我们将整个翻译、去字和文本检测流程全面升级为**当前最高质量的 100% 免费和本地 AI 技术方案**：

| 流水线组件 | 原版 / 其它分支 | 本免费 AI 分支 (This Fork) | 成本 |
| :--- | :--- | :--- | :---: |
| **智能翻译 (LLM)** | 依赖收费的 DeepL / OpenAI；容易遭遇速率限制与句意割裂 | **Google Gemini Flash & Lite** 通过内置 `gemini-proxy` 自动进行多密钥轮换与负载均衡。支持多气泡连贯拼接（Multi-bubble cohesion），漫画语境称谓对齐，100% 防幻觉。 | **¥0 (免费额度)** |
| **文字消除 (Inpaint)** | 渐变背景断层（gradient stepping）、明显拼接缝、像素渗色 | **Clean-Canvas LaMa Large** 结合 **拉普拉斯多频段金字塔融合 (Laplacian Multi-Band Pyramid Blending)**。平滑还原倾斜网点背景，非掩膜区域 100% 原图像素保真 (`Unmasked MAE = 0.000000`)。 | **¥0 (本地 GPU)** |
| **文本检测 (Detection)**| 漏检无框浮动文字，2K 高清图片缩放失真 | **ComicTextDetector 2K High-DPI (1536px)** 采用高灵敏度边缘阈值 (`text_thresh=0.25, link_thresh=0.20`)，完整捕获开放式气泡与悬浮文字。 | **¥0 (本地 GPU)** |
| **OCR (文字识别)** | 需配置繁琐第三方收费 API | **Windows Media OCR** (Windows 内置) + **PaddleOCR** 离线高速备选。 | **¥0 (离线)** |
| **字体排版 (Typography)** | 全局单字体单调填充 | **自动双字体嵌字系统**：普通对话与动作吼叫自动区分，完美支持多语言与越南语变音符号。 | **¥0 (内置)** |

---

### 🙏 致谢与上游原作者声明 (Credits & Upstream Attribution)

本项目是基于作者 [**@dmMaze**](https://github.com/dmMaze) 创立并主导的优秀开源项目 [**BallonsTranslator**](https://github.com/dmMaze/BallonsTranslator) 的增强分支。

- **原作者 / 上游仓库**：[dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator)
- **开源协议 (License)**：本项目严格遵循 [GNU General Public License v3.0 (GPL-3.0)](../LICENSE)，保留所有原始版权信息与声明。
- **分支目标**：由 [@kurrukado](https://github.com/kurrukado/BallonsTranslator) 维护，专注于 **100% 免费与本地 AI 流程**（Google Gemini 免费额度多密钥负载均衡、Clean-Canvas LaMa 去字、ComicTextDetector 2K 检测及多语言排版），让任何人都能以零 API 成本享受高质量漫画机翻嵌字体验。

原项目的所有基础 GUI 架构、画布渲染系统及核心算法均归原作者 [@dmMaze](https://github.com/dmMaze) 及上游贡献者所有。在此致以最诚挚的感谢！

---

<p align="center">
  <img src="src/ui0.jpg" alt="BalloonsTranslator 免费 AI 版界面">
</p>
<p align="center">
  <em>BalloonsTranslator 界面 (100% 免费 AI 模型版)</em>
</p>

---

# 主要功能 (Features)

* **一键全自动机翻**  
  - 自动完成文本检测、识别、去字、翻译并根据原文排版回填。
  - 译文回填参考对原文排版的估计，包括颜色、轮廓、角度、朝向、对齐方式与字号。
  - 完美支持日漫、韩漫与美漫。
  - 内置 `gemini-proxy`，自动负载均衡多个免费 Gemini API Key，彻底告别额度耗尽。

* **图像编辑与去字画笔**  
  - 支持掩膜编辑和修复画笔（类似 Photoshop 污点修复画笔）。
  - 原生适应条漫（Webtoon）等超长纵横比图像。

* **富文本编辑与排版**  
  - 所见即所得富文本编辑，支持多种字体预设与批量样式应用。
  - 支持全文/当前页文本查找与替换。
  - 支持将文本导入/导出到 Word 文档 (.docx)，便于校对协同。

---

# 安装与运行 (Installation)

## 环境要求
- **操作系统**: Windows 10/11 (64 位)。
- **Python**: 推荐 Python 3.10 至 3.12（或 Python 3.14 配合 frozen 模式运行）。
- **GPU**: 推荐配备支持 CUDA 的 NVIDIA 显卡以获得最佳去字与检测速度；亦支持 CPU 运行。

## 源码运行

1. **安装 Python 与 Git**:  
   下载并安装 [Python](https://www.python.org/downloads/)（记得勾选 Add Python to PATH）以及 [Git](https://git-scm.com/downloads)。

2. **克隆仓库**:
   ```bash
   git clone https://github.com/kurrukado/BallonsTranslator.git
   cd BallonsTranslator
   ```

3. **配置免费 Gemini API 密钥**:
   - 在 [Google AI Studio](https://aistudio.google.com/) 免费获取一个或多个 API Key。
   - 打开 `gemini-proxy/api-key.txt`（或从 `api-key.txt.example` 复制），每行填入一个 Key：
     ```text
     AIzaSyYourFirstGeminiApiKey...
     AIzaSyYourSecondGeminiApiKey...
     ```

4. **启动程序**:
   - **方式一（推荐）**: 双击运行 `start.bat`，自动拉起代理服务与主界面。
   - **方式二（命令行）**:
     ```bash
     python launch.py --frozen
     ```

---

# 使用说明 (Usage)

## 一键批量翻译文件夹
1. 启动程序后点击齿轮图标进入**设置面板**:
   - 选择翻译器：`gemini`（或 `gemini-proxy`）。
   - 选择源语言（Source Language）：例如 `Japanese` 或 `English`。
   - 选择目标语言（Target Language）：例如 `Chinese` 或 `Vietnamese`。
2. 点击文件夹图标，选择存放漫画图片的文件夹。
3. 点击 **Run** 按钮，静待程序自动完成检测、抹字与翻译回填。

<p align="center">
  <img src="src/run.gif" alt="一键翻译演示">
</p>

---

## 交互编辑工具

### 修复画笔
<p align="center">
  <img src="src/imgedit_inpaint.gif" alt="修复画笔">
</p>

### 矩形选框工具
<p align="center">
  <img src="src/rect_tool.gif" alt="矩形工具">
</p>

- 按下**鼠标左键**拖动矩形框抹除框内文字。
- 按下**鼠标右键**拉框清除修复结果，还原底层原始图像。
- 勾选“自动”拉完框立即修复，否则需按下空格键或修复按钮。按 `Ctrl+D` 可取消选框。

### 文本编辑与批量排版
<p align="center">
  <img src="src/textedit.gif" alt="文本编辑">
</p>

<p align="center">
  <img src="src/multisel_autolayout.gif" alt="批量排版">
</p>

---

# 常用快捷键 (Shortcuts)

| 快捷键 | 功能说明 |
| :--- | :--- |
| `Ctrl + Z` / `Ctrl + Y` | 撤销 / 重做 |
| `A` / `D` 或 `PageUp` / `PageDown` | 上一页 / 下一页（自动保存当前页面） |
| `T` | 切换到文本编辑模式 |
| `W` | 激活文本块创建模式（在画布右键拖拽生成新文本块） |
| `P` | 切换到画板/修复模式 |
| `Ctrl + (+ / -)` 或 `滚轮` | 缩放画布 |
| `Ctrl + A` | 选中当前页所有文本块 |
| `Ctrl + F` | 查找当前页文本 |
| `Ctrl + G` | 全局查找文本 |
| `Ctrl + B` / `Ctrl + I` / `Ctrl + U` | 加粗 / 斜体 / 下划线选中文本 |
| `Alt + 方向键` 或 `Alt + WASD` | 在各文本块之间快速切换 |
| `0 - 9` | 调节嵌字层与原图的透明度 |

---

# 命令行模式 (Headless CLI)

支持无 GUI 批量后台运行：
```bash
python launch.py --headless --exec_dirs "D:/Manga/Chapter_01,D:/Manga/Chapter_02"
```
所有参数（检测模型、翻译语言等）将自动从 `config/config.json` 载入。

---

# 开源协议 (License)

- 本项目基于 [GNU General Public License v3.0](../LICENSE) 开源。
- 本工具仅供学术研究、学习交流与汉化辅助用途，请尊重原作漫画版权。
