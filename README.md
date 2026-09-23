> [!IMPORTANT]  
> **Nếu bạn chia sẻ công khai kết quả dịch máy từ công cụ này mà chưa qua biên dịch hoặc hiệu đính bởi dịch giả có kinh nghiệm, vui lòng ghi rõ nguồn là bản dịch máy (machine translation).**

# 🎈 BalloonsTranslator — Low-Cost & Local AI Edition

> **Phiên bản BalloonsTranslator tối ưu hóa quy trình dịch thuật truyện tranh (Manga/Manhwa/Comic) bằng cách kết hợp các mô hình LLM chi phí thấp (Low-Cost Models) cùng các công nghệ AI chạy cục bộ (Local AI) hiệu năng cao.**

[![AI Architecture](https://img.shields.io/badge/Architecture-Low--Cost%20%26%20Local%20AI-brightgreen.svg)](#)
[![LLM Engine](https://img.shields.io/badge/LLM-Gemini%20Flash%20Series-blue.svg)](#)
[![Local Inpainting](https://img.shields.io/badge/Inpaint-LaMa%20Large%20(Local%20CUDA)-orange.svg)](#)
[![Text Detection](https://img.shields.io/badge/Detector-ComicTextDetector%202K-purple.svg)](#)
[![License](https://img.shields.io/badge/License-GPL--3.0-yellow.svg)](LICENSE)

[Tiếng Việt](README.md) | [English](README_EN.md) | [简体中文](doc/README_CN.md) | [日本語](doc/README_JA.md)

---

### 🌟 Giới thiệu Bản Phân Nhánh (About This Fork)

Bản phân nhánh này được phát triển nhằm tối ưu hóa chi phí vận hành và nâng cao chất lượng bản dịch cho dịch giả và các nhóm scanlation bằng cách kết hợp sức mạnh giữa mô hình ngôn ngữ lớn (LLM) chi phí thấp và các mô hình AI chuyên dụng chạy trực tiếp trên thiết bị (On-Device / Local AI):

- **Dịch thuật Ngữ cảnh Thông minh (Low-Cost LLM)**: Ứng dụng dòng mô hình **Google Gemini Flash & Flash-Lite** thông qua bộ điều phối `gemini-proxy` tự động cân bằng tải đa khóa (multi-key). Tích hợp cơ chế gộp chương (Chapter Batch Aggregation) hỗ trợ nối mạch câu đa bong bóng (multi-bubble cohesion), suy luận xưng hô scanlation tự nhiên theo vai vế nhân vật và khóa schema chống ảo giác.
- **Khôi phục Nền & Xóa chữ Cục bộ (Local Neural Inpainting)**: Sử dụng **Clean-Canvas LaMa Large** kết hợp thuật toán **Laplacian Multi-Band Pyramid Blending** chạy trực tiếp trên GPU/CPU, tái tạo mượt mà nền screentone dốc và bảo toàn toàn vẹn điểm ảnh gốc ngoài vùng mask (`Unmasked MAE = 0.000000`).
- **Nhận diện Vùng Chữ Tự Động (Local Text Detection)**: Sử dụng **ComicTextDetector 2K High-DPI (1536px)** với độ nhạy biên cao, bóc tách chính xác các khung thoại phức tạp và chữ rơi tự do ngoài bóng thoại.
- **Nhận diện Ký tự Đa Tầng Cục bộ (Local Multi-Tier OCR)**: Tích hợp **Windows Media OCR** (sẵn trên Windows) kết hợp **PaddleOCR** chạy ngoại tuyến, trích xuất chính xác chữ thoại truyện tranh.
- **Định dạng Phông chữ Bản quyền Scanlation**: Hệ thống tự động phân loại cảm xúc lời thoại để áp dụng bộ font chuyên dụng cho Manga/Comic tiếng Việt (`Yuki-CCMarianChurchlandJournal` cho thoại thường và `CCWildWordsRoman` cho gầm thét/hành động).

---

### 🙏 Lời Cảm Ơn & Nguồn Gốc Dự Án (Credits & Upstream Attribution)

Dự án này là bản phân nhánh (fork) cải tiến từ dự án mã nguồn mở tuyệt vời [**BallonsTranslator**](https://github.com/dmMaze/BallonsTranslator) được sáng lập và phát triển bởi tác giả [**@dmMaze**](https://github.com/dmMaze).

- **Tác giả gốc / Upstream Repository**: [dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator)
- **Giấy phép bản quyền (License)**: Dự án tuân thủ đầy đủ điều khoản của [GNU General Public License v3.0 (GPL-3.0)](LICENSE), giữ nguyên bản quyền và sự tôn trọng tuyệt đối với tác giả gốc.
- **Mục tiêu của Bản Fork**: Bản phân nhánh bởi [@kurrukado](https://github.com/kurrukado/BallonsTranslator) tập trung chuyên biệt vào việc kết hợp các mô hình ngôn ngữ chi phí thấp (Low-Cost LLM) cùng các công nghệ AI chạy cục bộ (Local AI on GPU/CPU) để mang lại giải pháp dịch truyện tranh hiệu quả, chất lượng cao và tiết kiệm chi phí cho dịch giả.

Mọi đóng góp cốt lõi về kiến trúc giao diện GUI, canvas rendering và các thuật toán nền tảng đều thuộc về tác giả [@dmMaze](https://github.com/dmMaze) cùng các contributor của dự án gốc. Xin chân thành cảm ơn tác giả!

---

<p align="center">
  <img src="doc/src/ui0.jpg" alt="Giao diện BalloonsTranslator">
</p>
<p align="center">
  <em>Giao diện BalloonsTranslator (Low-Cost & Local AI Edition)</em>
</p>

---

# Các tính năng nổi bật (Features)

* **Dịch thuật tự động 1-Click**  
  - Tự động nhận diện khung chữ, bóc tách văn bản, xóa chữ và điền văn bản dịch vào đúng vị trí.
  - Tự động ước lượng định dạng chữ gốc (màu sắc, viền, góc xoay, căn lề, kích thước) để dàn trang bản dịch tự nhiên nhất.
  - Tối ưu hóa đặc biệt cho Manga Nhật, Manhwa Hàn và Comic Âu Mỹ.
  - Tích hợp sẵn `gemini-proxy` tự động xoay tua nhiều API key Gemini, đảm bảo thông lượng dịch ổn định và tự động chuyển đổi mô hình (fallback chain).

* **Chỉnh sửa hình ảnh & Xóa chữ thủ công (Image Editing)**  
  - Hỗ trợ cọ xóa chữ (Inpainting Healing Brush) và chỉnh sửa mặt nạ (mask editing) chuyên nghiệp như Photoshop.
  - Xử lý mượt mà các trang truyện tranh dài (Webtoon) có tỷ lệ khung hình cực cao.

* **Biên tập văn bản trực quan (Text Editing)**  
  - Chỉnh sửa văn bản theo thời gian thực (WYSIWYG) với đầy đủ định dạng Rich Text.
  - Hỗ trợ [Text Style Presets](https://github.com/dmMaze/BallonsTranslator/pull/311) để lưu và áp dụng nhanh kiểu chữ yêu thích.
  - Tìm kiếm và thay thế toàn bộ từ khóa trong trang hoặc toàn bộ chương truyện.
  - Hỗ trợ xuất / nhập bản dịch qua tài liệu Word (.docx) để dịch giả bên ngoài hiệu đính dễ dàng.

---

# Hướng dẫn cài đặt (Installation)

## Yêu cầu hệ thống
- **Hệ điều hành**: Windows 10/11 (64-bit).
- **Python**: Khuyến nghị Python 3.10 đến 3.12 (hoặc Python 3.14 sử dụng chế độ frozen).
- **GPU (Card đồ họa)**: Khuyến nghị NVIDIA GPU (hỗ trợ CUDA) để tăng tốc độ xóa chữ và nhận diện; hỗ trợ chạy bằng CPU nếu không có GPU rời.

## Cài đặt từ mã nguồn (Chạy nhanh nhất)

1. **Cài đặt Python & Git**:  
   Tải và cài đặt [Python](https://www.python.org/downloads/) (chọn Add to PATH) và [Git](https://git-scm.com/downloads).

2. **Sao chép (Clone) kho mã nguồn**:
   ```bash
   git clone https://github.com/kurrukado/BallonsTranslator.git
   cd BallonsTranslator
   ```

3. **Cấu hình Gemini API Key**:
   - Truy cập [Google AI Studio](https://aistudio.google.com/) để lấy 1 hoặc nhiều API Key.
   - Mở file `gemini-proxy/api-key.txt` (hoặc copy từ `api-key.txt.example`) và dán các key vào, mỗi key trên một dòng:
     ```text
     AIzaSyYourFirstGeminiApiKey...
     AIzaSyYourSecondGeminiApiKey...
     ```

4. **Khởi chạy chương trình**:
   - **Cách 1 (Khuyên dùng)**: Nhấp đúp vào file `start.bat` để tự động bật proxy và giao diện dịch.
   - **Cách 2 (Dòng lệnh)**:
     ```bash
     python launch.py --frozen
     ```

---

# Hướng dẫn sử dụng (Usage)

## Dịch toàn bộ thư mục chỉ với 1 Click
1. Mở chương trình, bấm vào biểu tượng bánh răng **Cài đặt (Settings)**:
   - Chọn bộ dịch: `gemini` (hoặc `gemini-proxy`).
   - Chọn ngôn ngữ nguồn (Source Language): ví dụ `Japanese` (Tiếng Nhật) hoặc `English` (Tiếng Anh).
   - Chọn ngôn ngữ đích (Target Language): `Vietnamese` (Tiếng Việt).
2. Bấm vào biểu tượng thư mục để mở thư mục chứa các ảnh truyện cần dịch.
3. Nhấp nút **Run** (Chạy) và đợi chương trình hoàn tất toàn bộ các bước nhận diện, xóa chữ và điền lời dịch.

<p align="center">
  <img src="doc/src/run.gif" alt="Dịch tự động 1 click">
</p>

---

## Các công cụ chỉnh sửa thủ công

### Cọ xóa chữ (Inpaint Healing Brush)
<p align="center">
  <img src="doc/src/imgedit_inpaint.gif" alt="Cọ xóa chữ">
</p>

### Công cụ quét vùng chữ nhật (Rectangle Tool)
<p align="center">
  <img src="doc/src/rect_tool.gif" alt="Công cụ quét chữ nhật">
</p>

- Nhấn giữ **chuột trái** và kéo khung chữ nhật để xóa chữ bên trong khung.
- Nhấn giữ **chuột phải** và kéo khung để hủy kết quả xóa và phục hồi ảnh gốc.
- Tích chọn **Tự động (Auto)** để chương trình xóa chữ ngay khi nhả chuột, hoặc bấm phím `Space` (Cách) / nút `Inpaint` để xóa chữ. Bấm `Ctrl+D` để hủy khung chọn.

### Biên tập văn bản & Dàn trang tự động (Text Editing & Layout)
<p align="center">
  <img src="doc/src/textedit.gif" alt="Biên tập văn bản">
</p>

<p align="center">
  <img src="doc/src/multisel_autolayout.gif" alt="Dàn trang hàng loạt">
</p>

---

# Bảng phím tắt tiện lợi (Shortcuts)

| Phím tắt | Chức năng |
| :--- | :--- |
| `Ctrl + Z` / `Ctrl + Y` | Hoàn tác (Undo) / Làm lại (Redo) |
| `A` / `D` hoặc `PageUp` / `PageDown` | Chuyển trang trước / trang sau (tự động lưu trang hiện tại) |
| `T` | Chuyển sang chế độ Biên tập văn bản (Text Edit Mode) |
| `W` | Kích hoạt chế độ tạo khung chữ mới (sau đó kéo chuột phải trên canvas để vẽ khung) |
| `P` | Chuyển sang chế độ Cọ xóa chữ (Inpaint Brush Mode) |
| `Ctrl + (+ / -)` hoặc `Lăn chuột` | Phóng to / Thu nhỏ trang truyện |
| `Ctrl + A` | Chọn tất cả các khung chữ trên trang hiện tại |
| `Ctrl + F` | Tìm kiếm văn bản trên trang hiện tại |
| `Ctrl + G` | Tìm kiếm văn bản trên toàn bộ các trang (Global Find) |
| `Ctrl + B` / `Ctrl + I` / `Ctrl + U` | In đậm / In nghiêng / Gạch chân văn bản đang chọn |
| `Alt + Phím mũi tên` hoặc `Alt + WASD` | Chuyển nhanh giữa các khung thoại |
| `0 - 9` | Điều chỉnh độ trong suốt của lớp dịch / ảnh gốc |

---

# Chế độ dòng lệnh không cần giao diện (Headless CLI)

Bạn có thể chạy dịch hàng loạt qua dòng lệnh mà không cần mở giao diện đồ họa:
```bash
python launch.py --headless --exec_dirs "D:/Manga/Chapter_01,D:/Manga/Chapter_02"
```
Toàn bộ cài đặt (mô hình nhận diện, ngôn ngữ dịch) sẽ được nạp tự động từ tệp `config/config.json`.

---

# Giấy phép & Tuyên bố miễn trừ trách nhiệm (License)

- Dự án được phát hành theo giấy phép [GNU General Public License v3.0](LICENSE).
- Công cụ được tạo ra nhằm mục đích học tập, nghiên cứu công nghệ AI và hỗ trợ cộng đồng dịch giả. Vui lòng tôn trọng bản quyền của các tác giả truyện tranh gốc.
