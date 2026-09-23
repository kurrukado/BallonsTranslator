> [!IMPORTANT]  
> **本ツールの機械翻訳結果を公開する場合、経験豊富な翻訳者による校正がない場合は、機械翻訳であることを明記してください。**

# 🎈 BalloonsTranslator — 低コスト＆ローカル AI 最適化版 (Low-Cost & Local AI Edition)

> **低コストな大規模言語モデル（Low-Cost LLMs）と高性能なオンデバイス AI 技術を組み合わせた BalloonsTranslator の強化フォーク版。Google Gemini Flash シリーズのマルチキー負荷分散、Clean-Canvas LaMa オンデバイス画像修復、ComicTextDetector 2K 高解像度検出、多言語・ベトナム語組版プリセットを統合。**

[![AI Architecture](https://img.shields.io/badge/Architecture-Low--Cost%20%26%20Local%20AI-brightgreen.svg)](#)
[![LLM Engine](https://img.shields.io/badge/LLM-Gemini%20Flash%20Series-blue.svg)](#)
[![Local Inpainting](https://img.shields.io/badge/Inpaint-LaMa%20Large%20(Local%20CUDA)-orange.svg)](#)
[![Text Detection](https://img.shields.io/badge/Detector-ComicTextDetector%202K-purple.svg)](#)
[![License](https://img.shields.io/badge/License-GPL--3.0-yellow.svg)](../LICENSE)

[日本語](README_JA.md) | [English](../README_EN.md) | [Tiếng Việt](../README.md) | [简体中文](README_CN.md)

---

### 🌟 本フォークの特徴 (About This Fork)

本フォークは、高コストパフォーマンスな言語モデルとオンデバイス AI 技術を連携させ、翻訳者やコミック翻訳グループ向けに高品質かつ運用コストを抑えたワークフローを提供します：

- **文脈認識スマート翻訳 (Low-Cost LLM)**：内蔵 `gemini-proxy` により **Google Gemini Flash & Flash-Lite** を自動マルチキー分散。複数フキダシの一括文脈結合（Multi-bubble cohesion）、漫画の自然な代名詞・語尾処理、ハルシネーション防止スキーマを統合。
- **オンデバイス文字消去 (Local Neural Inpainting)**：**Clean-Canvas LaMa Large** と **ラプラシアン・マルチバンド・ピラミッド・ブレンディング** をローカル GPU/CPU 上で直接実行。スクリーントーンの傾斜も滑らかに復元し、マスク外の元画像ピクセルを 100% 保持 (`Unmasked MAE = 0.000000`)。
- **高解像度テキスト検出 (Local AI)**：**ComicTextDetector 2K High-DPI (1536px)** の高感度エッジ検出により、複雑なフキダシや枠外の浮遊文字を確実に捕捉。
- **ローカルオフライン OCR**：Windows 標準の **Windows Media OCR** と **PaddleOCR** オフラインエンジンを統合し、外部依存なしに素早くテキストを認識。
- **漫画専用タイポグラフィ組版**：通常会話と叫び・アクションを自動識別し、最適な漫画フォントスタイルを自動適用。

---

### 🙏 謝辞と上流オリジナルリポジトリへの帰属 (Credits & Upstream Attribution)

本プロジェクトは、[**@dmMaze**](https://github.com/dmMaze) 氏が作成・主導する素晴らしいオープンソースプロジェクト [**BallonsTranslator**](https://github.com/dmMaze/BallonsTranslator) の機能強化フォークです。

- **原作者 / アップストリーム**：[dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator)
- **ライセンス (License)**：[GNU General Public License v3.0 (GPL-3.0)](../LICENSE) に厳格に準拠し、原著作権表記を保持しています。
- **フォークの目的**：[@kurrukado](https://github.com/kurrukado/BallonsTranslator) により、低コスト LLM とオンデバイス AI 処理の効率的な統合に注力し、実用的かつ高品質な漫画翻訳環境を提供します。

GUI アーキテクチャ、キャンバスレンダリング、コアアルゴリズムのすべての功績は、原作者 [@dmMaze](https://github.com/dmMaze) 氏およびアップストリームのコントリビューターに帰属します。心より感謝申し上げます！

---

<p align="center">
  <img src="src/ui0.jpg" alt="BalloonsTranslator UI">
</p>
<p align="center">
  <em>BalloonsTranslator UI (Low-Cost & Local AI Edition)</em>
</p>

---

# 主な特徴 (Features)

* **ワンクリック全自動翻訳**  
  - テキスト検出・認識・消去・翻訳・組版回填までをワンストップで実行。
  - 原文の書式（フォント色、縁取り、角度、配置、サイズ）を自動推定して忠実にレイアウト。
  - 日本のマンガ、韓国のウェブトゥーン、海外コミックに対応。
  - 内蔵の `gemini-proxy` により複数の Gemini API キーを自動ローテーションし、安定したスループットと柔軟なモデルフォールバックを確保。

* **画像編集と消去ブラシ**  
  - マスク編集および修復ブラシ（Photoshop のスポット修復ブラシに類似）を搭載。
  - 縦長スクロールの Webtoon など極端なアスペクト比の画像にも対応。

* **リッチテキスト編集と一括組版**  
  - WYSIWYG エディタによる直感的なテキスト修正。
  - [テキストスタイルプリセット](https://github.com/dmMaze/BallonsTranslator/pull/311) による素早いスタイル適用。
  - ページ内および全ページのテキスト一括検索・置換。
  - Word 文書 (.docx) へのインポート/エクスポートに対応し、外部での校正作業もスムーズ。

---

# インストールと起動 (Installation)

## 動作環境
- **OS**: Windows 10/11 (64-bit)。
- **Python**: Python 3.10 〜 3.12 推奨（Python 3.14 の場合は `--frozen` オプションで動作可能）。
- **GPU**: CUDA 対応 NVIDIA GPU 推奨（CPU のみでも動作可能）。

## ソースコードからの実行（推奨）

1. **Python & Git のインストール**:  
   [Python](https://www.python.org/downloads/)（Add Python to PATH にチェック）および [Git](https://git-scm.com/downloads) をインストール。

2. **リポジトリのクローン**:
   ```bash
   git clone https://github.com/kurrukado/BallonsTranslator.git
   cd BallonsTranslator
   ```

3. **Gemini API キーの設定**:
   - [Google AI Studio](https://aistudio.google.com/) で 1 つ以上の API キーを取得。
   - `gemini-proxy/api-key.txt`（または `api-key.txt.example` をコピー）を開き、1 行に 1 つずつキーを貼り付け：
     ```text
     AIzaSyYourFirstGeminiApiKey...
     AIzaSyYourSecondGeminiApiKey...
     ```

4. **アプリケーションの起動**:
   - **方法 1（推奨）**: `start.bat` をダブルクリック（プロキシと GUI が自動起動します）。
   - **方法 2（コマンドライン）**:
     ```bash
     python launch.py --frozen
     ```

---

# 使い方 (Usage)

## フォルダの一括自動翻訳
1. アプリを起動し、**設定アイコン**（歯車）をクリック：
   - 翻訳エンジン: `gemini`（または `gemini-proxy`）。
   - 翻訳元言語（Source）: `Japanese` または `English`。
   - 翻訳先言語（Target）: `Vietnamese` や `English` など。
2. **フォルダアイコン** をクリックして漫画画像の入ったフォルダを選択。
3. **Run** ボタンをクリックすると、検出・消去・翻訳・組版が全自動で進行します。

<p align="center">
  <img src="src/run.gif" alt="ワンクリック自動翻訳">
</p>

---

## インタラクティブ編集ツール

### 修復ブラシ (Inpaint Brush)
<p align="center">
  <img src="src/imgedit_inpaint.gif" alt="修復ブラシ">
</p>

### 矩形消去ツール (Rectangle Tool)
<p align="center">
  <img src="src/rect_tool.gif" alt="矩形ツール">
</p>

- **左クリック** ドラッグで矩形内の文字を消去。
- **右クリック** ドラッグで消去を取り消し、元画像を復元。
- 「自動」にチェックを入れるとマウスを離した瞬間に修復を実行します。または `Space` キーで修復、`Ctrl+D` で選択解除。

### テキスト編集と自動レイアウト
<p align="center">
  <img src="src/textedit.gif" alt="テキスト編集">
</p>

<p align="center">
  <img src="src/multisel_autolayout.gif" alt="一括レイアウト">
</p>

---

# 便利なショートカット一覧 (Shortcuts)

| ショートカット | 機能 |
| :--- | :--- |
| `Ctrl + Z` / `Ctrl + Y` | 元に戻す (Undo) / やり直す (Redo) |
| `A` / `D` または `PageUp` / `PageDown` | 前のページ / 次のページ（自動保存） |
| `T` | テキスト編集モードに切り替え |
| `W` | 新規テキストブロック作成モード（キャンバス右ドラッグで枠作成） |
| `P` | 修復ブラシモードに切り替え |
| `Ctrl + (+ / -)` または `マウスホイール` | キャンバスの拡大 / 縮小 |
| `Ctrl + A` | ページ内のすべてのテキストブロックを選択 |
| `Ctrl + F` | ページ内テキスト検索 |
| `Ctrl + G` | 全ページ横断テキスト検索 |
| `Ctrl + B` / `Ctrl + I` / `Ctrl + U` | 太字 / 斜体 / 下線 |
| `Alt + 矢印キー` または `Alt + WASD` | テキストブロック間の移動 |
| `0 - 9` | 翻訳レイヤーと元画像の透明度調整 |

---

# CLI モード (Headless)

GUI を起動せずにコマンドラインから一括翻訳を実行できます：
```bash
python launch.py --headless --exec_dirs "D:/Manga/Chapter_01,D:/Manga/Chapter_02"
```
すべてのパラメータは `config/config.json` から読み込まれます。

---

# ライセンス (License)

- [GNU General Public License v3.0](../LICENSE) のもとで公開されています。
- 本ツールは学術研究・AI 技術検証・ファン翻訳支援を目的としています。原作者の著作権を尊重してご使用ください。
