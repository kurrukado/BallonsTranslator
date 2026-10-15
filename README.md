> [!IMPORTANT]  
> **Nếu sử dụng, vui lòng ghi rõ đây là bản dịch máy (machine translation). Tool này được làm hoàn toàn bởi AI nên sẽ có sai sót, bản fork này chỉ support mạnh cho dịch sang Tiếng Việt**

# BalloonsTranslator — Low-Cost & Local AI Edition
[![License](https://img.shields.io/badge/License-GPL--3.0-yellow.svg)](LICENSE)

[Tiếng Việt](README.md) | [English](README_EN.md) | [简体中文](doc/README_CN.md) | [日本語](doc/README_JA.md)

---

### Giới thiệu Bản Phân Nhánh (About This Fork)

Bản phân nhánh này nhằm tối ưu chi phí và nâng cao chất lượng bản dịch bằng mô hình chi phí thấp và local:

- **Dịch thuật Ngữ cảnh**: sử dụng **Google Gemini Flash & Flash-Lite** và cơ chế gộp chương, suy luận xưng hô scanlation tự nhiên theo vai vế nhân vật và khóa schema chống ảo giác.
- **Khôi phục Nền & Xóa chữ Cục bộ**: Sử dụng **Clean-Canvas LaMa Large** kết hợp thuật toán **Laplacian Multi-Band Pyramid Blending** chạy trực tiếp trên GPU/CPU.
- **Nhận diện Vùng Chữ Tự Động (Local Text Detection)**: Sử dụng **ComicTextDetector 2K High-DPI (1536px)** với độ nhạy biên cao, bóc tách chính xác các khung thoại phức tạp và chữ rơi tự do ngoài bóng thoại.
- **Nhận diện Ký tự Đa Tầng Cục bộ (Local Multi-Tier OCR)**: Tích hợp **Windows Media OCR** (sẵn trên Windows) kết hợp **PaddleOCR**
- **Font chữ có sẵn**: Yuki-CCMarianChurchlandJournal và CCWildWordsRoman (nguồn từ nhóm trên facebook).
---

### 🙏 Lời Cảm Ơn & Nguồn Gốc Dự Án (Credits & Upstream Attribution)

Mọi đóng góp cốt lõi về kiến trúc giao diện GUI, canvas rendering và các thuật toán nền tảng đều thuộc về tác giả [@dmMaze](https://github.com/dmMaze) cùng các contributor của dự án gốc. Xin chân thành cảm ơn tác giả!

---

# Hướng dẫn cài đặt (Installation)

## Yêu cầu hệ thống
- **Hệ điều hành**: Windows 10/11 (64-bit).
- **Python**: Khuyến nghị Python 3.10 đến 3.12 (hoặc Python 3.14 sử dụng chế độ frozen).
- **GPU (Card đồ họa)**: Khuyến nghị NVIDIA GPU (hỗ trợ CUDA) để tăng tốc độ xóa chữ và nhận diện; có hỗ trợ chạy bằng CPU.

## Cài đặt từ mã nguồn (Chạy nhanh nhất)

1. **Cài đặt Python & Git**:  
   Tải và cài đặt [Python](https://www.python.org/downloads/) (chọn Add to PATH) và [Git](https://git-scm.com/downloads).

2. **Sao chép (Clone) kho mã nguồn**:
   ```bash
   git clone https://github.com/kurrukado/BallonsTranslator.git
   cd BallonsTranslator
   ```

3. **Khởi chạy chương trình**:
   - **Cách 1 (Khuyên dùng)**: Nhấp đúp vào file `start.bat` để tự động bật proxy và giao diện dịch.
   - **Cách 2 (Dòng lệnh)**:
     ```bash
     python launch.py --frozen
     ```

---

# Giấy phép & Tuyên bố miễn trừ trách nhiệm (License)

- Dự án được phát hành theo giấy phép [GNU General Public License v3.0](LICENSE).
- Công cụ được tạo ra nhằm mục đích học tập, nghiên cứu công nghệ AI và hỗ trợ cộng đồng dịch giả. Vui lòng tôn trọng bản quyền của các tác giả truyện tranh gốc.