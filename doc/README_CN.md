> [!IMPORTANT]  
> **如打算公开分享本工具的机翻结果，且没有有经验的译者进行过完整的翻译或校对，请在显眼位置注明机翻。**

# 🎈 BalloonsTranslator — 低成本与本地 AI 优化版 (Low-Cost & Local AI Edition)

> **结合低成本大语言模型（Low-Cost LLMs）与高性能本地 AI 技术的 BalloonsTranslator 分支版本。集成 Google Gemini Flash 系列多密钥负载均衡、Clean-Canvas LaMa 本地修复、ComicTextDetector 2K 高清文本检测及多语言/越南语汉化排版预设。**

[![AI Architecture](https://img.shields.io/badge/Architecture-Low--Cost%20%26%20Local%20AI-brightgreen.svg)](#)
[![LLM Engine](https://img.shields.io/badge/LLM-Gemini%20Flash%20Series-blue.svg)](#)
[![Local Inpainting](https://img.shields.io/badge/Inpaint-LaMa%20Large%20(Local%20CUDA)-orange.svg)](#)
[![Text Detection](https://img.shields.io/badge/Detector-ComicTextDetector%202K-purple.svg)](#)
[![License](https://img.shields.io/badge/License-GPL--3.0-yellow.svg)](../LICENSE)

[简体中文](README_CN.md) | [English](../README_EN.md) | [Tiếng Việt](../README.md) | [日本語](README_JA.md)

---

### 🌟 分支架构特点 (About This Fork)

本分支致力于通过融合高性价比语言模型与本地端侧 AI 技术，为个人译者与汉化组提供兼顾高质量与低成本的完整漫画翻译工具链：

- **语境智能翻译 (Low-Cost LLM)**：采用 **Google Gemini Flash & Flash-Lite** 模型，通过内置 `gemini-proxy` 自动执行多密钥轮换与请求分发。支持章节级聚合翻译（Chapter Batch Aggregation），实现多气泡语意平滑衔接、角色语境称谓对齐及严格防幻觉校验。
- **本地端侧图像消除 (Local Neural Inpainting)**：基于 **Clean-Canvas LaMa Large** 与 **拉普拉斯多频段金字塔融合 (Laplacian Multi-Band Pyramid Blending)** 在本地 GPU/CPU 直接运算，平滑修复网点背景并保持非掩膜区域 100% 原始像素完整度 (`Unmasked MAE = 0.000000`)。
- **高分辨率文本检测 (Local AI)**：采用 **ComicTextDetector 2K High-DPI (1536px)** 高灵敏度边缘算法，完整检出各类复杂对话框与无框浮动文字。
- **多级本地离线 OCR (Local OCR)**：内置 **Windows Media OCR** 与 **PaddleOCR** 离线引擎，快速精准提取原文。
- **专业级字体排版系统**：自动识别常规对话与动作音效/吼叫，自动套用适配漫画风格的专用字体样式。

---

### 🙏 致谢与上游原作者声明 (Credits & Upstream Attribution)

本项目是基于作者 [**@dmMaze**](https://github.com/dmMaze) 创立并主导的优秀开源项目 [**BallonsTranslator**](https://github.com/dmMaze/BallonsTranslator) 的增强分支。

- **原作者 / 上游仓库**：[dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator)
- **开源协议 (License)**：本项目严格遵循 [GNU General Public License v3.0 (GPL-3.0)](../LICENSE)，保留所有原始版权信息与声明。
- **分支目标**：由 [@kurrukado](https://github.com/kurrukado/BallonsTranslator) 维护，专注于低成本大模型（Low-Cost LLM）与本地端侧 AI 算力的高效整合，为漫画翻译本地化提供经济、高效且高质量的生产力工具。

原项目的所有基础 GUI 架构、画布渲染系统及核心算法均归原作者 [@dmMaze](https://github.com/dmMaze) 及上游贡献者所有。在此致以最诚挚的感谢！

---

<p align="center">
  <img src="src/ui0.jpg" alt="BalloonsTranslator 界面">
</p>
<p align="center">
  <em>BalloonsTranslator 界面 (Low-Cost & Local AI Edition)</em>
</p>

---

# 主要功能 (Features)

* **一键全自动机翻**  
  - 自动完成文本检测、识别、去字、翻译并根据原文排版回填。
  - 译文回填参考对原文排版的估计，包括颜色、轮廓、角度、朝向、对齐方式与字号。
  - 完美支持日漫、韩漫与美漫。
  - 内置 `gemini-proxy`，自动负载均衡多个 Gemini API Key，保障翻译吞吐与顺畅的模型回退机制。

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

3. **配置 Gemini API 密钥**:
   - 在 [Google AI Studio](https://aistudio.google.com/) 获取一个或多个 API Key。
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
