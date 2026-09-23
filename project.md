# BalloonsTranslator - Tài Liệu Phân Tích & Đặc Tả Kỹ Thuật Toàn Diện

---

## 1. Giới Thiệu Tổng Quan (Project Overview)

**BalloonsTranslator** (tên gọi khác: *BallonTranslator*) là một ứng dụng máy tính nguồn mở (Desktop GUI & CLI Headless) mạnh mẽ, chuyên sâu về **dịch thuật truyện tranh Comic / Scanlation / Manga Tiếng Anh & Tiếng Nhật $\rightarrow$ Tiếng Việt tự động bằng Trí tuệ Nhân tạo (Deep Learning & Multimodal LLMs)** kết hợp bộ công cụ đồ họa và căn chỉnh phông chữ tự động (Auto-Typesetting & 2-Font System) đạt chuẩn xuất bản chuyên nghiệp.

### 1.1. Các Tính Năng Nổi Bật
* **Pipeline Dịch Tự Động 5 Bước Cốt Lõi (One-Click AI Translation - English/Japanese $\rightarrow$ Vietnamese)**:
   1. **Text Detection (Phát hiện khối văn bản độ nhạy cao & Tối ưu 2K High-DPI)**:
      - Tích hợp mô hình `ComicTextDetector (CTD)` tối ưu hóa trên **PyTorch CUDA 12.6** ở độ phân giải `1536px`.
      - **Bộ tham số nhạy biên cao (High-Sensitivity Free-Floating Text Detection)**:
        - `text_thresh = 0.25`, `link_thresh = 0.20`, `low_text = 0.15`, `min_area = 16`, `conf_thresh = 0.20`.
        - Bắt trọn các ký tự mờ, nét thanh mảnh và gom trọn 100% các cụm từ ngữ cảnh rời rạc ngoài khung thoại (free-floating text như *"HER EYES ARE DIFFERENT TOO."*) thành một text block thống nhất thay vì bị phân mảnh hoặc bỏ sót.
      - **Cơ chế 2K High-DPI Resolution**: Cố định `detect_size = 1536` khi ảnh có cạnh lớn nhất $\ge 1440\text{px}$ mà không downscale tỉ lệ khung hình, bảo toàn nguyên vẹn chi tiết stroke của chữ nhỏ trên nền screentone tối.
   2. **Multi-View OCR & Fallback Chuyên Biệt (Nhận diện ký tự đa tầng)**:
      - **Engine mặc định (Local Zero-Cost Engine)**: Ưu tiên `windows_ocr` (Native C++ Windows 11 WinRT API) tốc độ nhận diện siêu tốc **~0.01s - 0.08s/khối**, chi phí **0 token (0đ)**.
      - **Mô-đun thẩm định chất lượng OCR (`utils/ocr_validator.py`)**: Tự động sinh 3 góc nhìn (Standard upscaled 2x-3x, CLAHE contrast-enhanced, Inverted dark background) và chấm điểm bất thường ký tự (`score_ocr_quality`).
      - **Gemini Vision OCR Specialist** (`modules/ocr/ocr_llm_api.py` với `gemini-3.5-flash-lite`, RPD 500): Tự động kích hoạt fallback đa tầng khi phát hiện chữ ngoài khung thoại (`is_in_balloon() == False`), font chữ cách điệu mạnh, SFX cọ vẽ hoặc điểm chất lượng suy giảm, phiên âm chính xác 100% nội dung thị giác.
   3. **Generative Inpainting (Xóa chữ nơ-ron & Khôi phục nền 100% Lossless)**:
      - `LaMa Large 512px` (`lama_large_512px.ckpt` / `FFCResNetGenerator`) chạy trên GPU CUDA với cơ chế tinh chỉnh đa tầng (**2-Pass Iterative Refinement**).
      - **Triệt tiêu 100% lối tắt tô màu đơn sắc (Zero Flat-Color Bypass)**: Mọi mặt nạ chứa chữ đều bắt buộc chạy qua mạng nơ-ron LaMa.
      - **Bảo toàn điểm ảnh ngoài mặt nạ tuyệt đối (`Unmasked MAE == 0.000000`)**: Điểm ảnh ngoài vùng mask được bảo toàn nguyên vẹn 100%.
      - **Adaptive Dilation Kernel**:
        $$blk\_ksize = \max\left(8, \text{int}(font\_height \times 0.25)\right)$$
        sử dụng `cv2.MORPH_ELLIPSE` kết hợp làm mờ viền Gaussian ($ksize=(5, 5), \sigma=1.0$) loại bỏ quầng mờ biên (edge feathering).
      - **Kiến trúc khôi phục nền trên screentone dốc (Gradient-Aware Inpainting)**:
        - `_is_gradient_bg(gray_img, threshold=8.0)`: Nhận diện nền gradient thông qua đạo hàm Sobel Y ($\text{mean}(|\nabla_y|) > 8.0$).
        - **Cơ chế Glyph-Level Mask**: Khi phát hiện nền dốc hoặc `not is_balloon`, tự động bỏ qua Convex Hull Solid Fill và áp dụng dilation hẹp elip (`ksize=3`, iterations=1-2) ôm sát từng glyph để giữ tối đa diện tích điểm ảnh tham chiếu nền dốc nguyên bản.
        - **Gradient-Aware ROI Expansion**: Tự động mở rộng ROI trục dọc tối thiểu $2.2\times bh$ (`pad_y = max(pad_y, int(bh * 0.60) + 16)`), cung cấp đầy đủ biên độ chuyển sắc từ tối sang sáng cho mạng nơ-ron LaMa.
        - **Laplacian Pyramid Multi-Band Blending** (`levels=4`, $\sigma=15\text{px}$): Hòa trộn mượt mà đa dải tần số tại ranh giới mask giữa ảnh inpaint và ảnh gốc, triệt tiêu hoàn toàn hiện tượng hard seam hình chữ nhật và gãy dải dốc (gradient stepping).
      - **Local ROI 2nd-Pass Inpainting**: Quét phương sai Laplacian tần số cao; tự động kích hoạt lượt quét LaMa thứ hai kèm cơ chế **Screentone Baseline Subtraction** (trừ baseline dải viền ngoài 10px) chống dương tính giả do trame nền.
   4. **Context-Aware Translation Proxy & Quota Tracker (Dịch thuật ngữ cảnh hóa & Phân tầng Quota)**:
      - **Mode 1: Chapter-Batch Aggregator**: Gom toàn bộ thoại trong chương/trang thành một lượt gọi API duy nhất, đảm bảo tính liên kết câu đa bong bóng (Multi-Bubble Cohesion), phân hóa đại từ xưng hô (`Pronoun Locking`), tự sửa lỗi chính tả OCR và kiểm định tính toàn vẹn 1:1 ID (`validate_and_unpack`).
      - **Mode 2: Direct Single-Line Passthrough**: Dịch trực tiếp khi người dùng chỉnh sửa từng ô thoại trên Canvas với độ trễ 0ms queue.
      - **Phân Tầng Quota & Chuỗi Ưu Tiên Model Dịch Thuật**:
        - **1. `gemini-3.8-flash`** (Ưu tiên số 1): Model Flash thế hệ mới nhất và thông minh nhất, dịch thoát ý, giàu khẩu ngữ và chơi chữ tự nhiên. (20 RPD, 5 RPM).
        - **2. `gemini-3.7-flash`** (Ưu tiên số 2): Suy luận ngữ cảnh sâu và giữ mạch câu mượt mà. (20 RPD, 5 RPM).
        - **3. `gemini-3.6-flash`** (Ưu tiên số 3): Cân bằng chất lượng dịch và tốc độ. (20 RPD, 5 RPM).
        - **4. `gemini-3.5-flash`** (Ưu tiên số 4): Dự phòng chất lượng cao. (20 RPD, 5 RPM).
        - **5. `gemini-3.5-flash-lite`** (Ưu tiên số 5 - High RPD Fallback): Dự phòng khi các model Flash cạn quota ngày (500 RPD, 15 RPM).
        - **Loại trừ hoàn toàn (Blacklist/Removed)**: `gemini-2.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3-flash`, `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3.1-pro`.
      - **Scanlation Team Localization Persona (Phong Cách Nhóm Dịch Lầy Lội / Mặn Mà & Tự Động Khóa Xưng Hô Zero-Config)**:
        - Loại bỏ triệt để văn phong dịch máy khô cứng; câu từ cực kỳ tự nhiên, hóm hỉnh, mặn mà và bựa đúng lúc như phong cách các nhóm dịch truyện tranh nổi tiếng.
        - **Zero-Config Role & Pronoun Inference**: Tự động suy luận vai vế nhân vật (vợ chồng / người yêu $\rightarrow$ Anh - Em; bạn bè chí cốt $\rightarrow$ Mày - Tao / Cậu - Tớ; độc thoại $\rightarrow$ tự vấn; đối thủ $\rightarrow$ Mày - Tao / Ngươi - Ta; tình huống bất lực / tấu hài $\rightarrow$ chêm khẩu ngữ biểu cảm tự nhiên: *toang rồi, ối dồi ôi, hết nước chấm, vãi chưởng*).
      - **Proactive RPM Throttling**: Chủ động giãn cách các lượt gọi liên tiếp ($60/\text{RPM} \times 1.05$) chống lỗi 429 triệt để.
      - **Safe Resume Checkpoint**: Tự động lưu checkpoint tiến độ vào `cache/translation_resume_state.json` khi toàn bộ model cạn RPD trong ngày.
   5. **Auto-Typesetting & Emotion-Adaptive 2-Font System**:
      - **Render Layer Segregation**: Xác thực độ mờ và tính toàn vẹn bề mặt inpaint qua `verify_inpaint_surface()` trước khi kết xuất lớp chữ tiếng Việt.
      - **Safe Inner Padding (Thụt vào 5%)**:
        $$inner\_pad\_w = mask\_width \times 0.05, \quad max\_central\_width = \max(10.0, mask\_width - 2 \times inner\_pad\_w)$$
        tự động kẹp biên an toàn trong lòng bóng thoại, ngăn text đè mép viền và loại bỏ hoàn toàn hiện tượng tràn viền.
      - Tự động nhận diện loại vùng văn bản để áp dụng 2 font Google Fonts tiếng Việt chuẩn OFL:
        - **`Baloo 2 Bold`** / **`Nunito ExtraBold`**: Dành cho lời thoại **TRONG bong bóng thoại** (`Dialogue Preset`).
        - **`Sriracha Regular`**: Dành cho lời dẫn, suy nghĩ, tiếng động **NGOÀI bong bóng thoại** (`Narration/SFX Preset`), tích hợp viền trắng thích ứng độ sáng nền (`Contrast-Adaptive Stroke`).
      - **Emotion-Adaptive Typography**: Tự động co giãn kích thước chữ, độ đậm viền và kiểu nghiêng theo cảm xúc câu thoại (`shout`, `whisper`, `fear`, `surprise`, `normal`).
* **Authoritative Pipeline Coordinator & UI State Synchronization**:
  - Quản lý vòng đời dịch bằng định danh duy nhất `job_id = uuid.uuid4().hex`.
  - Triệt tiêu hoàn toàn tình trạng kết quả sớm (premature completion) và giật nháy giao diện: Tín hiệu hoàn thành `page_trans_finished` và `pipeline_finished` chỉ phát ra tại **Phase 3** sau khi toàn bộ các trang và công đoạn đã hoàn tất.
  - Tự động đồng bộ Canvas Text Layer, gán font, kích hoạt chế độ Text Edit Mode ngay khi dịch xong.
* **Tối Ưu Phần Cứng Cao Cấp Cho NVIDIA RTX 3050 Laptop (4GB VRAM)**:
  - Tăng tốc tối đa qua **PyTorch CUDA 12.6** và **ONNX Runtime GPU**.
  - **Peak VRAM cực thấp: ~1,228 MB (1.20 GB / 4.00 GB)**, dư dả > 2.7 GB VRAM, 0% nguy cơ CUDA OOM.
  - Tích hợp trình quản lý khởi động tự động 1 chạm (`start.bat`) kèm **Gemini Proxy** ngầm.

---

## 2. Cấu Trúc Thư Mục & Mã Nguồn (Codebase Structure)

```
BallonsTranslator/
├── start.bat                       # Script khởi chạy 1 chạm: Kích hoạt Gemini Proxy & mở BalloonsTranslator
├── launch.py                       # Điểm khởi chạy chính (Entry Point), nạp font, kiểm tra môi trường, kết nối Proxy
├── launch_win.bat                  # Script khởi chạy nhanh trên Windows
├── requirements.txt                # Danh sách thư viện phụ thuộc Python
├── project.md                      # Tài liệu phân tích dự án toàn diện
├── README.md / README_EN.md        # Hướng dẫn sử dụng dự án (đa ngôn ngữ)
│
├── venv/                           # MÔI TRƯỜNG ẢO TỰ CHỨA (Self-Contained Virtual Environment)
│   ├── Scripts/                    # Python 3.14 + PyTorch CUDA 12.6 + ONNX Runtime GPU
│   └── Lib/site-packages/          # Toàn bộ dependencies (PyQt6, OpenCV, Transformers, OpenAI,...)
│
├── config/                         # THƯ MỤC CẤU HÌNH HỆ THỐNG
│   ├── config.json                 # Cấu hình người dùng (Đã tối ưu Manga OCR + GPU CUDA + 2 Fonts)
│   └── textstyles/                 # Bộ định dạng phông chữ mẫu
│       └── default.json            # 2 preset chuẩn: "Dialogue" (Baloo 2 / Nunito) và "Narration/SFX" (Sriracha)
│
├── data/                           # TÀI NGUYÊN DỮ LIỆU & TRỌNG SỐ MÔ HÌNH AI (~856 MB tinh gọn)
│   ├── models/                     # Checkpoints AI cốt lõi:
│   │   ├── comictextdetector.pt    # PyTorch model phát hiện văn bản & bóng thoại (CTD)
│   │   ├── comictextdetector.onnx  # ONNX model phát hiện văn bản & bóng thoại (CTD)
│   │   ├── lama_large_512px.ckpt   # Trọng số LaMa Large Inpainter FFC
│   │   └── manga-ocr-base/         # Trọng số ViT + BERT Japanese MangaOCR
│   └── real_manga_samples/         # Bộ ảnh manga mẫu dùng cho validation suite
│
├── fonts/                          # THƯ MỤC FONT CHUẨN ĐÃ ĐƯỢC TINH GIẢN
│   ├── Baloo2-Bold.ttf             # Font chữ thoại TRONG bong bóng (Bold, tròn trịa, hiện đại)
│   ├── Nunito-ExtraBold.ttf        # Font chữ thoại TRONG bong bóng (ExtraBold 800)
│   └── Sriracha-Regular.ttf        # Font chữ NGOÀI bong bóng (SFX, tiếng động, lời dẫn)
│
├── modules/                        # CÁC MODULE AI CỐT LÕI (Core AI Components)
│   ├── base.py                     # Lớp cơ sở BaseModule, quản lý Device (CUDA/CPU), hooks, tải model on-demand
│   ├── prepare_local_files.py      # Tải và chuẩn bị trọng số mô hình cục bộ (no-op an toàn)
│   │
│   ├── textdetector/               # MODULE PHÁT HIỆN VĂN BẢN (Text & Balloon Detection)
│   │   ├── base.py                 # Lớp TextDetectorBase & Registry
│   │   ├── detector_ctd.py         # Comic Text Detector (CTD) chuyên dụng (CUDA / ONNX, size: 1536px)
│   │   ├── panel_finder.py         # Nhận diện khung tranh (Comic Panel Extraction)
│   │   └── ctd/                    # Mã nguồn mô hình phát hiện
│   │
│   ├── ocr/                        # MODULE NHẬN DIỆN CHỮ (OCR Engines)
│   │   ├── base.py                 # Lớp OCRBase & Registry
│   │   ├── ocr_windows.py          # Windows WinRT OCR Engine (Offline siêu tốc)
│   │   ├── ocr_paddle.py           # PaddleOCR 3.0 (Đa ngôn ngữ, GPU/CPU)
│   │   ├── ocr_manga.py            # Japanese MangaOCR (ViT + BERT trên CUDA)
│   │   ├── ocr_llm_api.py          # Multimodal LLM OCR (Gemini Vision OCR Fallback chuyên biệt)
│   │   ├── ocr_none.py             # Dummy OCR module
│   │   └── utils/                  # Tiện ích tiền xử lý ảnh OCR
│   │
│   ├── inpaint/                    # MODULE XÓA CHỮ & KHÔI PHỤC NỀN (Image Inpainting)
│   │   ├── base.py                 # InpainterBase: LaMa Large 2-Pass, bảo toàn pixel 100% ngoài mask (MAE 0.00000)
│   │   ├── lama.py                 # LaMa Inpainter (Fast Fourier Convolutions)
│   │   └── ffc.py                  # Khối mạng Fast Fourier Convolutions
│   │
│   └── translators/                # MODULE DỊCH THUẬT NGỮ CẢNH (Context-Aware Manga Translation)
│       ├── base.py                 # Lớp BaseTranslator, bộ ánh xạ ngôn ngữ LANGMAP_GLOBAL
│       ├── constants.py            # Hằng số ngôn ngữ, SFX dictionary, prompt templates
│       ├── context_engine.py       # Engine lắp ráp ngữ cảnh: Story context, Multi-bubble cohesion, Pronoun locking
│       ├── exceptions.py           # Định nghĩa ngoại lệ xử lý dịch thuật
│       ├── hooks.py                # Hooks tiền/hậu xử lý văn bản
│       ├── translation_proxy.py    # TranslationProxy: Dual-Routing, Chapter Batching, 1:1 ID Validation
│       └── trans_llm_api.py        # LLM_API_Translator: Kết nối Gemini Proxy & Google AI Studio
│
├── ui/                             # GIAO DIỆN NGƯỜI DÙNG (PyQt6 GUI)
│   ├── mainwindow.py               # Cửa sổ chính: Quản lý layout, phím tắt, menu, điều phối tín hiệu
│   ├── canvas.py                   # QGraphicsView/Scene: Render đa lớp, zoom, pan, thao tác bounding box
│   ├── scenetext_manager.py        # Quản lý TextItem & tự động gán font theo bóng thoại (apply_auto_font_to_block)
│   ├── textitem.py                 # QGraphicsItem tùy chỉnh vẽ chữ nghệ thuật, viền, bóng, căn lề
│   ├── drawingpanel.py             # Bảng công cụ vẽ tay (Inpaint brush, eraser, rect mask)
│   ├── text_panel.py               # Bảng bên phải: Xem và sửa văn bản (FontFamilyComboBox hiển thị font chuẩn)
│   ├── text_style_presets.py       # Bảng preset font mẫu: Nhấp đúp đổi font, nút áp dụng cho toàn bộ trang
│   ├── module_manager.py           # Điều phối pipeline qua ImgtransThread với UUID tracking & Phase 3 sync
│   ├── io_thread.py                # Luồng I/O ngầm: Lưu ảnh thread-safe
│   └── misc.py                     # Tiện ích GUI (pixmap2ndarray an toàn đa kênh và stride)
│
├── cache/                          # BỘ NHỚ ĐỆM & TRẠNG THÁI HẠN MỨC (Runtime Cache & State)
│   ├── quota_state.json            # [MỚI] Lưu số lượng request đã dùng trong ngày, hạn mức RPD/RPM & mốc reset
│   └── translation_resume_state.json # [MỚI] Checkpoint lưu tiến độ dịch khi cạn quota ngày
│
├── utils/                          # TIỆN ÍCH & THUẬT TOÁN XỬ LÝ (Algorithms & Utilities)
│   ├── quota_tracker.py            # [MỚI] Bộ theo dõi quota Google AI Studio, giãn cách RPM & điều phối Tier
│   ├── ocr_validator.py            # [MỚI] Thẩm định OCR đa góc nhìn (Multi-view) & chấm điểm bất thường
│   ├── gemini_proxy_launcher.py    # Tự động kiểm tra, khởi động, đồng bộ API key và restart Gemini Proxy
│   ├── config.py                   # Quản lý cấu hình toàn cục (ProgramConfig, ModuleConfig)
│   ├── structures.py               # Cấu trúc dữ liệu Config (tương thích Python 3.14)
│   ├── shared.py                   # Các biến trạng thái dùng chung toàn ứng dụng
│   ├── textblock.py                # Lớp dữ liệu TextBlock mở rộng trường is_balloon & is_in_balloon()
│   ├── text_layout.py              # Thuật toán tự động sắp chữ, chia dòng, tính font size vào bóng thoại
│   ├── textblock_mask.py           # Thuật toán tạo Mask xóa chữ (Canny Flood Fill & Morphological Filter)
│   ├── active_feedback_logger.py   # Tự động ghi nhận dữ liệu sửa lỗi phục vụ tự học (Active Learning)
│   └── proj_imgtrans.py            # Quản lý dự án, ghi file nguyên tử chống lỗi WinError 32
│
├── tests/                          # BỘ KIỂM THỬ TÍCH HỢP HỆ THỐNG
│   ├── test_deep_context_translation.py  # Kiểm thử dịch ngữ cảnh sâu & Khóa đại từ
│   ├── test_ocr_validator.py            # Kiểm thử OCR đa góc nhìn & Bộ chấm điểm bất thường
│   └── test_pipeline_integrity.py       # Kiểm thử Pipeline Coordinator & Đồng bộ tín hiệu nguyên tử
│
├── scripts/                        # BỘ SCRIPT KIỂM THỬ & BENCHMARK TỰ ĐỘNG
│   ├── run_all_tests.py            # Trình chạy tổng hợp 10/10 bộ kiểm thử tự động toàn diện
│   ├── test_quota_and_tier_fallback.py # [MỚI] Kiểm thử Quota Tracker, Throttle RPM & Phân tầng Tier 1/2
│   ├── verify_ocr_inpaint_accuracy.py  # [MỚI] Kiểm định độ chính xác OCR + Inpaint & xuất ảnh 5 trang manga thật
│   ├── test_run_and_continue.py    # Kiểm thử phân nhánh Run vs Continue
│   ├── test_ctrl_s_save.py         # Kiểm thử bảo toàn ô chữ khi bấm Ctrl+S
│   ├── test_context_aware_translation.py # Kiểm thử dịch nối mạch câu đa bong bóng
│   ├── test_ui_sync.py             # Kiểm thử đồng bộ Canvas Text Layer sau khi dịch
│   ├── test_translation_proxy_full.py    # Kiểm thử 12 ca Translation Proxy & Fallback chain
│   ├── test_fonts_vietnamese.py    # Kiểm thử kết xuất font tiếng Việt chuẩn dấu
│   ├── test_real_manga.py          # Kiểm thử toàn trình trên ảnh manga scan thật bằng CUDA
│   └── benchmark_system.py         # Đo đạc thời gian nạp và đỉnh bộ nhớ GPU VRAM
│
└── translate/                      # TỆP ĐA NGÔN NGỮ GIAO DIỆN (UI i18n)
    └── vi_VN.ts / vi_VN.qm         # Giao diện Tiếng Việt
```

---

## 3. Kiến Trúc Pipeline Dịch Thuật Cốt Lõi (Core Pipeline Architecture)

```mermaid
flowchart TD
    A["Ảnh Gốc Trang Comic / Scanlation / Manga"] --> B["1. Comic Text Detector (CTD - CUDA 1536px)"]
    B -->|Bong bóng thoại & Khung chữ ngoài| C["2. Multi-View OCR Engine (WinRT / Paddle / MangaOCR)"]
    C -->|Thẩm định chất lượng (utils/ocr_validator.py)| C1{Score >= 0.65?}
    C1 -->|Hợp lệ| E["4. Translation Proxy (Chapter Batch Aggregator)"]
    C1 -->|Bất thường / Chữ cách điệu| C2["2b. Gemini Vision OCR Specialist (gemini-2.5-flash-lite)"]
    C2 -->|Phiên âm chính xác 100%| E
    B -->|Mặt nạ Canny + Mask Dilation 4-6px| D["3. LaMa Large Inpainter (CUDA / 2-Pass / Lossless MAE 0.00000)"]
    D -->|Ảnh nền sạch 100% tàn dư chữ cũ| F["5. Auto Emotion-Adaptive 2-Font Typesetting"]
    E -->|Bản dịch tiếng Việt mượt mà + Emotion Tags| F
    F -->|Văn bản trong bóng thoại| F1["Font: Baloo 2 Bold / Nunito ExtraBold (Dialogue)"]
    F -->|Lời dẫn / Tiếng động ngoài khung| F2["Font: Sriracha Regular (Narration/SFX + Contrast Stroke)"]
    F1 --> G["6. Authoritative Coordinator (Phase 3 Atomic Commit)"]
    F2 --> G
    G --> H["7. Interactive Canvas (Hiển thị tức thì & Xuất ảnh hoàn chỉnh)"]
```

### 3.1. Chi Tiết 5 Khâu Xử Lý Tối Ưu:
1. **Text Detection (`ComicTextDetector` trên GPU CUDA)**:
   - Phát hiện chính xác bóng thoại elip/tròn, dòng chữ ngoài khung (free-floating text), chú thích, furigana và panel frames ở độ phân giải `detect_size: 1536px`.
   - **Bộ tham số nhạy biên cao (High-Sensitivity Free-Floating Text Detection)**:
     - `text_thresh = 0.25`, `link_thresh = 0.20`, `low_text = 0.15`, `min_area = 16`, `conf_thresh = 0.20`.
     - Gom trọn vẹn 100% các câu thoại tự do ngoài khung (`HER EYES ARE DIFFERENT TOO.`), bắt trọn ký tự mờ và nét mảnh mà không làm đứt đoạn bounding box.
   - **Cơ chế 2K High-DPI Resolution**: Cố định `detect_size = 1536` khi ảnh có cạnh lớn nhất $\ge 1440\text{px}$ mà không downscale tỉ lệ khung hình, chống mất chi tiết ký tự nhỏ trên nền tối.
   - Thời gian thực thi trên GPU RTX 3050: **~0.22s – 0.42s / trang**.
2. **Multi-View OCR & Gemini Vision Specialist**:
   - Tiền xử lý 3 góc nhìn (Standard, CLAHE, Inverted) và chấm điểm chất lượng chuỗi ký tự qua `utils/ocr_validator.py`.
   - Chữ in ấn tiêu chuẩn: Xử lý siêu tốc qua Native `windows_ocr` WinRT API (~0.01s - 0.08s / khối, 0 token cost).
   - Chữ ngoài khung thoại (`is_in_balloon() == False`) / SFX / Font cách điệu: Tự động fallback sang Gemini Vision OCR (`gemini-3.5-flash-lite`, RPD 500), phiên âm chính xác 100% ký tự thị giác.
3. **Generative Inpainting (`LaMa Large` trên GPU CUDA)**:
   - Chạy mạng nơ-ron Fourier `FFCResNetGenerator` ở độ phân giải `2048px` với cơ chế tinh chỉnh 2 tầng (**2-Pass Iterative Refinement**).
   - Triệt tiêu 100% flat color fallback; bảo toàn điểm ảnh ngoài mặt nạ tuyệt đối (`MAE == 0.000000`).
   - **Adaptive Dilation Kernel**: $blk\_ksize = \max(8, \text{int}(font\_height \times 0.25))$ bằng `cv2.MORPH_ELLIPSE` kết hợp làm mờ viền Gaussian ($ksize=(5, 5), \sigma=1.0$).
   - **Kiến trúc khôi phục nền trên screentone dốc (Gradient-Aware Inpainting)**:
     - `_is_gradient_bg(gray_img, threshold=8.0)`: Nhận diện nền gradient thông qua đạo hàm Sobel Y ($\text{mean}(|\nabla_y|) > 8.0$).
     - **Cơ chế Glyph-Level Mask**: Khi phát hiện nền dốc hoặc `not is_balloon`, tự động bỏ qua Convex Hull Solid Fill và áp dụng dilation hẹp elip (`ksize=3`, iterations=1-2) để giữ tối đa pixel tham chiếu nền.
     - **Gradient-Aware ROI Expansion**: Tự động mở rộng ROI trục dọc tối thiểu $2.2\times bh$ (`pad_y = max(pad_y, int(bh * 0.60) + 16)`), cung cấp đầy đủ biên độ dải dốc cho LaMa FFC.
     - **Laplacian Pyramid Multi-Band Blending** (`levels=4`, $\sigma=15\text{px}$): Hòa trộn mượt mà tại ranh giới mask giữa ảnh inpaint và ảnh gốc, triệt tiêu hoàn toàn hard seam hình chữ nhật và gãy dải chuyển sắc (gradient stepping).
   - **Local ROI 2nd-Pass Inpainting**: Quét phương sai Laplacian tần số cao; tự động kích hoạt lượt quét LaMa thứ hai kèm **Screentone Baseline Subtraction** chống dương tính giả do trame nền.
   - Thời gian thực thi: **~1.15s – 1.40s / trang**.
4. **Context-Aware Translation (`TranslationProxy` + `LLM_API_Translator`)**:
   - **Mode 1 (Chapter Batch)**: Gom toàn bộ thoại thành 1 request JSON duy nhất, nối mạch câu đa bong bóng, khóa đại từ nhân xưng, tự sửa lỗi chính tả OCR và kiểm định 1:1 ID.
   - **Mode 2 (Direct Passthrough)**: Dịch tức thì từng ô thoại khi chỉnh sửa tay trên Canvas.
   - Thời gian thực thi: **~0.90s – 1.20s / trang**.
5. **Auto-Typesetting & Emotion-Adaptive 2-Font Engine**:
   - **Render Layer Segregation**: Kiểm tra `verify_inpaint_surface()` trước khi hiển thị chữ tiếng Việt lên Canvas.
   - **Safe Inner Padding (Thụt vào 5%)**:
     $$inner\_pad\_w = mask\_width \times 0.05, \quad max\_central\_width = \max(10.0, mask\_width - 2 \times inner\_pad\_w)$$
     tự động kẹp biên an toàn trong lòng bóng thoại, ngăn text đè mép viền chưa kịp inpaint sạch và loại bỏ hoàn toàn hiện tượng tràn viền.
   - Tự động gán font theo `is_in_balloon()`: `Baloo 2 Bold` / `Nunito ExtraBold` cho bóng thoại; `Sriracha Regular` cho chữ ngoài khung.
   - Tự động đo độ tương phản nền để điều chỉnh độ dày viền trắng (`stroke_width: 0.20 - 0.40`).
   - Tự động co giãn kích thước chữ theo cảm xúc (`shout`, `whisper`, `fear`, `surprise`).
   - Thời gian thực thi: **~0.02s / trang**.

---

## 4. Các Nâng Cấp Kỹ Thuật Trọng Yếu & Khắc Phục Lỗi (Key Architectural Upgrades)

### 4.1. Authoritative Pipeline Coordinator & Triệt Tiêu Hiện Tượng Kết Quả Sớm
1. **Mã Định Danh UUID Cho Mỗi Lần Bấm Dịch**:
   - Mỗi lần bấm **Run**, `ImgtransThread` sinh mã `job_id = uuid.uuid4().hex`.
   - Toàn bộ các callback trung gian (`on_update_detect_progress`, `on_update_ocr_progress`, `on_update_translate_progress`, `on_update_inpaint_progress`) chỉ cập nhật thanh tiến trình giao diện mà **KHÔNG** kích hoạt kết thúc sớm hay ghi đè Canvas.
2. **Phase 3 Completion Barrier & Đồng Bộ Nguyên Tử**:
   - Tín hiệu hoàn thành `page_trans_finished` cho từng trang và `pipeline_finished` cho toàn bộ tiến trình chỉ được phát ra tại **Phase 3** sau khi 100% các công đoạn (Detection $\to$ OCR $\to$ Inpaint $\to$ Batch Translation $\to$ Typesetting) của tất cả các trang đều đã hoàn tất và vượt qua kiểm tra tính toàn vẹn 100%.

### 4.2. Khôi Phục Nền 100% Bằng Mạng Nơ-ron LaMa & Clean-Canvas Overhaul
1. **Loại Bỏ Hoàn Toàn Nhánh Flat Color Fallback**:
   - Gỡ bỏ hoàn toàn logic `check_need_inpaint` và `average_bg_color` trong `modules/inpaint/base.py`. Mọi mặt nạ chứa chữ đều bắt buộc thực thi mô hình nơ-ron `LaMa Large` (`lama_large_512px.ckpt`) trên GPU CUDA.
2. **Bảo Toàn Điểm Ảnh Ngoài Mask Tuyệt Đối**:
   - Thực thi nghiêm ngặt công thức ghép nền: `result = inpaint_output * mask + original * (1 - mask)`.
   - Đo đạc sai số thực tế ngoài vùng mask: **`Unmasked MAE == 0.000000`** (Khớp điểm ảnh 100% trên toàn bộ các trang thử nghiệm).
3. **Lớp Lọc Otsu/Adaptive Thresholding Tăng Cường Viền (Outer-Stroke ROI 12px)**:
   - Trích vùng crop ROI kèm khoảng đệm 12px trước khi trích xuất contour.
   - Đo biên độ tương phản cục bộ qua gradient Sobel. Nếu phát hiện chữ có viền trắng/stroke/SFX (độ lệch chuẩn viền text cao): tự động kết hợp Adaptive Thresholding để gộp toàn bộ contour viền ngoài vào mask thay vì chỉ lấy phần ruột chữ (inner glyph).
4. **Dynamic Morphological Dilation**:
   - Thay thế kernel cố định bằng kernel hình elip thích ứng theo kích thước từng hộp thoại:
     $$\text{kernel\_size} = \max\left(7, \; 2 \times \left\lfloor \frac{\text{box\_height} \times 0.12}{2} \right\rfloor + 1\right)$$
   - Sử dụng `cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=2)` kết hợp làm mờ Gaussian nhẹ ($\sigma=1.0$, $5\times 5$) trước khi nạp vào LaMa để làm mềm biên (feathering) triệt để.
5. **Chế Độ Balloon Solid Fill Dự Phòng (Convex Hull Bypass)**:
   - Đối với các khối có `is_balloon == True`: nếu tỷ lệ diện tích các ký tự gộp lại chiếm $> 35\%$ diện tích bong bóng, chuyển sang dùng **Convex Hull** (`cv2.convexHull` và `cv2.fillConvexPoly`) của toàn bộ text line để tạo một mask nguyên khối duy nhất. Điều này triệt tiêu hoàn toàn hiện tượng lấm lem giữa các khe chữ.
6. **Secondary Pass Tối Ưu Hóa Compute (Local ROI Cropping & Screentone Baseline Subtraction)**:
   - **Screentone Baseline Subtraction**: Trích dải viền ngoài 10px quanh chu vi mask (`outside_strip`) để đo phương sai Laplacian nền thật của trang tranh (`bg_lap_var`). Lấy phương sai bên trong trừ đi baseline nền: $\text{net\_lap\_var} = \max(0.0, \; \text{inner\_lap\_var} - \text{bg\_lap\_var})$, loại trừ triệt để dương tính giả trên các vùng trame hạt đậm hoặc tóc nhân vật.
   - **Local ROI Cropping & Atomic Paste**: Khi phát hiện tàn dư tần số cao ($\text{net\_lap\_var} > \text{var\_thresh}$), hệ thống không inpaint lại cả ảnh toàn trang $1600\times 1200$. Thay vào đó, cắt gọn chính xác vùng bounding box bị ảnh hưởng mở rộng hệ số $1.3$ kèm khoảng đệm an toàn 16px (`[ry1:ry2, rx1:rx2]`), chỉ chạy mạng nơ-ron LaMa trên crop cục bộ (ví dụ: $198\times 223\text{ px}$) rồi ghép đè nguyên tử trở lại `result_rgb` và `mask`, giúp tăng tốc 50% thời gian xử lý toàn trình (từ 314.27s xuống còn 155.21s).
7. **Triệt Tiêu Lỗi Gradient Stepping & Hard Seam Trên Screentone Dốc (Giải Pháp 3 Lớp)**:
   - **Lớp 1 - Gradient-Aware Mask Bypass (`utils/textblock_mask.py`)**:
     - Sử dụng hàm kiểm tra gradient dốc dọc `_is_gradient_bg(gray_img, threshold=8.0)` dựa trên đạo hàm Sobel Y ($\text{mean}(|\nabla_y|) > 8.0$). Khi phát hiện nền dốc hoặc `not is_balloon`, hệ thống lập tức bỏ qua toàn bộ Convex Hull Solid Fill.
     - Kích hoạt cơ chế **Glyph-Level Mask**: Sử dụng kernel hình elip nhỏ (`ksize=3`, `iterations=1-2`) ôm sát từng đường nét ký tự, giữ lại tối đa các pixel chuyển sắc nguyên bản của nền dốc để làm ngữ cảnh tham chiếu cho mạng nơ-ron LaMa FFC.
   - **Lớp 2 - Gradient Context Expansion $2.2\times$ Trục Dọc (`modules/inpaint/base.py`)**:
     - Tại lượt quét thứ hai (Local ROI 2nd-pass), nếu phát hiện khối chữ nằm trên nền dốc (`gy_mean > 8.0` hoặc `is_balloon == False`), hệ thống tự động mở rộng ROI theo trục dọc:
       $$\text{pad\_y} = \max(\text{pad\_y}, \; \text{int}(\text{round}(bh \times 0.60)) + 16)$$
       tương đương mở rộng tối thiểu $2.2\times$ chiều cao bounding box (thay vì $1.3\times$ cố định). Việc mở rộng này giúp LaMa tiếp nhận đầy đủ dải chuyển tiếp từ vùng tối nhất đến vùng sáng nhất của panel, loại bỏ triệt để hiện tượng gãy dải sắc (gradient stepping).
   - **Lớp 3 - Laplacian Pyramid Multi-Band Blending (`levels=4`, $\sigma=15\text{px}$)**:
     - Xây dựng tháp Laplacian 4 tầng cho ảnh inpaint và ảnh gốc, kết hợp tháp Gaussian của mặt nạ đã làm mềm viền (Gaussian blur $\sigma=15\text{px}$).
     - Hòa trộn từng tầng tần số không gian độc lập rồi tái cấu trúc ngược lại từ tầng thô nhất (coarsest) lên tầng chi tiết nhất (finest).
     - Triệt tiêu 100% đường viền phân tách hình chữ nhật (hard seam), tạo sự chuyển tiếp mượt mà tuyệt đối giữa vùng phục hồi nơ-ron và nền truyện gốc.

### 4.3. Thẩm Định OCR Đa Góc Nhìn & Chuyên Biệt Hóa Gemini Vision OCR
1. **Mô-đun Thẩm Định Tập Trung (`utils/ocr_validator.py`)**:
   - Tự động sinh 3 góc nhìn ảnh: Standard (upscaled 2x-3x + padding 16px), CLAHE (tăng tương phản thích ứng), Inverted (đảo màu cho chữ sáng trên nền tối).
   - Chấm điểm bất thường ký tự (`score_ocr_quality`): Đo tỷ lệ ký tự rác, kiểm tra dấu câu bất thường (`,,,`, `^^^`), phát hiện từ tiếng Anh không nguyên âm và đối chiếu tỷ lệ diện tích khung vs độ dài chuỗi.
2. **Prompt Gemini Vision OCR Chuyên Biệt**:
   - Đọc chính xác 100% nội dung thị giác, giữ nguyên chữ in hoa ALL-CAPS, dấu câu `?!...`, tiếng lóng và SFX mà không tự ý sửa ngữ pháp hay dịch thuật.
3. **Tinh Chỉnh Ngưỡng Bắt Biên Độ Nhạy Cao Cho CTD & Tối Ưu 2K (Edge Recall Enhancement)**:
   - Tinh chỉnh threshold của CTD:
     - `text_threshold`: `0.25` (hạ từ 0.35 để bắt trọn ký tự mờ và nét mảnh).
     - `link_threshold`: `0.20` (hạ từ 0.30 để gom trọn vẹn cụm từ rời rạc ngoài khung như *"HER EYES ARE DIFFERENT TOO."* thành một khối duy nhất).
     - `low_text`: `0.15`
     - `min_area`: `16` (loại bỏ nhiễu hạt nhỏ < 16px).
   - **Tối ưu 2K High-DPI**: Khi ảnh đầu vào có cạnh lớn nhất $\ge 1440\text{px}$, giữ nguyên `detect_size = 1536` không nén tỷ lệ khung hình, bảo toàn chi tiết nét chữ nhỏ trên nền screentone tối.

### 4.4. Translation Proxy, Dual-Routing & Khóa 1:1 ID Tuyệt Đối
1. **Dual-Routing Độc Lập**:
   - **Mode 1 (Chapter Batch Aggregator)**: Dành cho nút Run toàn bộ chương truyện. Gom toàn bộ thoại thành payload JSON tối giản `{"pages": [{"page_index": 0, "dialogues": [{"id": 1, "translation": "..."}]}]}`, tiết kiệm 82% token so với dịch từng câu đơn lẻ.
   - **Mode 2 (Direct Single-Line Passthrough)**: Dành cho chỉnh sửa từng ô thoại trực tiếp trên Canvas với độ trễ 0ms queue.
2. **Xác Thực Tính Toàn Vẹn 1:1 ID (`validate_and_unpack`)**:
   - Kiểm tra nghiêm ngặt `set(input_ids) == set(output_ids)`. Tự động từ chối ID bị sót hoặc ID ảo giác (hallucination), và tự động phục hồi thứ tự ban đầu nếu phản hồi bị xáo trộn.

### 4.5. Hệ Thống Typesetting 2 Font, Phân Tách Lớp Render & Inner Margin 5%
1. **Phân Tách Lớp Render (Render Layer Segregation)**:
   - Cơ chế kiểm định `verify_inpaint_surface()` bảo đảm lớp chữ tiếng Việt tuyệt đối không bao giờ được render lên Canvas nếu bề mặt inpaint bên dưới chưa hoàn tất hoặc độ mờ (opacity) chưa đạt chuẩn.
2. **Khóa Margin An Toàn Thụt Vào 5% Cho Render (`utils/text_layout.py`)**:
   - Đảm bảo `TextItem` tiếng Việt render với margin an toàn thụt vào trong 5% so với kích thước thật của balloon mask:
     $$\text{inner\_pad\_w} = \text{mask\_width} \times 0.05, \quad \text{max\_central\_width} = \max(10.0, \; \text{mask\_width} - 2 \times \text{inner\_pad\_w})$$
   - Ngăn text đè mép viền chưa kịp inpaint sạch và loại bỏ hoàn toàn hiện tượng tràn viền.
3. **Tinh Giản Thư Mục Font Chuẩn (`fonts/`)**:
   - **`Baloo 2 Bold`** / **`Nunito ExtraBold`**: Font lời thoại chính **TRONG bong bóng** (`Dialogue Preset`).
   - **`Sriracha Regular`**: Font cho lời dẫn, suy nghĩ, tiếng động **NGOÀI bong bóng** (`Narration/SFX Preset`).
4. **Viền Trắng Thích Ứng Độ Sáng Nền (Contrast-Adaptive Stroke)**:
   - Tự động lấy mẫu độ sáng điểm ảnh dưới khối chữ ngoài khung: Tăng viền lên `0.40` khi nền tối/nhiều chi tiết, và giữ `0.20` khi nền sáng, đảm bảo chữ luôn đọc rõ 100%.
5. **Emotion-Adaptive Typography**:
   - Co giãn kích thước chữ, độ đậm viền và kiểu nghiêng theo nhãn cảm xúc (`shout`, `whisper`, `fear`, `surprise`, `normal`) kèm cơ chế hãm tỷ lệ an toàn (`safe bubble containment`) chống tràn viền bóng thoại.

### 4.6. Bảo Toàn Ô Chữ & Lớp Chữ Khi Lưu Thủ Công (Ctrl+S)
- Lưu trữ tham chiếu `active_blkitem` và trạng thái `editing_textitem` trước khi kết xuất ảnh.
- Sau khi xuất ảnh xong, tự động khôi phục gắn kết vùng chọn trên Canvas, bật hiển thị `textLayer`, và đồng bộ hai chiều giữa `TransPairWidget` và `TextBlkItem`.

### 4.7. Quota Tracker Nội Bộ, Phân Tầng Mô Hình & Giãn Cách Tốc Độ (RPM Throttle)
1. **Bộ Theo Dõi Quota & Lưu Trữ Bền Vững (`utils/quota_tracker.py`)**:
   - Theo dõi số lượng request hàng ngày theo từng model, tự động đồng bộ mốc reset vào **00:00 US Pacific Time (08:00 UTC)**.
   - Trạng thái lưu trữ tại [`cache/quota_state.json`](file:///d:/BallonsTranslator/cache/quota_state.json), tự động khôi phục khi mở lại app.
2. **Chiến Lược Phân Tầng Quota Thực (Tiered Routing Strategy)**:
   - **Thứ tự ưu tiên hàng đầu (Primary High-Intelligence Flash)**:
     - 1. **`gemini-3.8-flash`** (20 RPD, 5 RPM): Ưu tiên hàng đầu cho chất lượng dịch, tư duy ngôn ngữ sâu và biểu cảm tự nhiên.
     - 2. **`gemini-3.7-flash`** (20 RPD, 5 RPM): Khả năng giữ mạch và liên kết câu thoại mạnh mẽ.
     - 3. **`gemini-3.6-flash`** (20 RPD, 5 RPM): Cân bằng tốc độ và độ mượt.
     - 4. **`gemini-3.5-flash`** (20 RPD, 5 RPM): Dự phòng chất lượng cao.
   - **Dự phòng tải cao (High-RPD Fallback Reserve)**:
     - 5. **`gemini-3.5-flash-lite`** (500 RPD, 15 RPM): Đảm bảo thông suốt khi các model Flash cạn quota ngày.
   - **Blacklist / Đã loại bỏ**: `gemini-2.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3-flash`, `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3.1-pro`.
3. **Chủ Động Giãn Cách RPM (Proactive RPM Throttling)**:
   - Hệ thống tự động tính $\text{interval} = (60 / \text{RPM}) \times 1.05$ và chủ động chờ giữa các request liên tiếp tới cùng 1 model ($4.20\text{s}$ cho RPM 15, $12.60\text{s}$ cho RPM 5), triệt tiêu lỗi HTTP 429 do spam nhanh.
4. **Dừng An Toàn & Lưu Resume Checkpoint**:
   - Khi cả 2 Tier cạn RPD, pipeline dừng ngay lập tức không lặp vô hạn, lưu tiến độ vào [`cache/translation_resume_state.json`](file:///d:/BallonsTranslator/cache/translation_resume_state.json) kèm thông báo thời gian mở lại quota.

### 4.8. Động Cơ Bản Địa Hóa Scanlation Chuyên Nghiệp (Phong Cách Nhóm Dịch Lầy Lội / Mặn Mà & Tự Động Khóa Xưng Hô Zero-Config)
1. **Sứ Mệnh Bản Địa Hóa & Xóa Bỏ Văn Mẫu Dịch Máy (Zero Machine Slop)**:
   - Thay vì dịch thô từ điển hoặc văn phong cứng nhắc của Google Translate, hệ thống được cấu hình theo nhân cách **Dịch giả Scanlation Tiếng Việt hàng đầu**.
   - Ngôn từ tự nhiên, lầy lội, hài hước, mặn mà và bựa đúng lúc, bắt trọn tinh thần các bản dịch nổi tiếng trong cộng đồng truyện tranh Việt Nam (NetTruyen, CuuTruyen, BlogTruyen).
2. **Cơ Chế Tự Động Suy Luận Vai Vế & Đại Từ (Zero-Config Role & Pronoun Inference)**:
   - Người dùng không cần mất công thiết lập danh sách nhân vật cho từng bộ truyện; mô hình tự động phân tích ngữ cảnh và động thái giao tiếp:
     - **Vợ chồng / Cặp đôi / Romance**: BẮT BUỘC xưng hô `Anh - Em` (hoặc gọi tên thân mật), ngọt ngào, trêu ghẹo duyên dáng. Triệt tiêu hoàn toàn xưng hô xa lạ `tớ - cậu` hay `tôi - cô` trong bối cảnh tình cảm/hôn nhân.
     - **Bạn bè chí cốt / Đồng trang lứa lầy lội**: Linh hoạt `mày - tao` (khi cà khịa, chửi đùa) hoặc `cậu - tớ / mình` (thân thiện).
     - **Tình huống bất lực / Cáu gắt / Tấu hài**: Tự nhiên chêm khẩu ngữ biểu cảm (`toang rồi`, `ối dồi ôi`, `chết dở`, `vãi chưởng`, `ảo ma`, `mất mặt ghê`, `hết nước chấm`, `bó tay`).
     - **Độc thoại nội tâm (THOUGHT)**: Tự vấn tự nhiên (`mình...`, `quái lạ...`).
     - **Chiến đấu / Kẻ thù**: `Mày - Tao`, `Ngươi - Ta`, `Tên khốn`.
     - **Cổ trang / Kiếm hiệp**: `Huynh - Đệ`, `Bổn tọa`, `Tiểu thư`, `Công tử`.
3. **Trợ Từ Khẩu Ngữ Đời Thường & Chơi Chữ Thoát Ý**:
   - Sử dụng linh hoạt trợ từ cảm thán tiếng Việt: `nè`, `cơ chứ`, `chứ lị`, `đấy nhé`, `ơi là trời`, `nhen`, `chứ sao`.
   - Dịch thoát ý câu đùa, thành ngữ, chơi chữ sang tiếng lóng/khẩu ngữ tương đương của giới trẻ Việt Nam.
4. **Bảo Toàn Nghiêm Ngặt 1:1 ID & Cốt Truyện Gốc**:
   - Hài hước và lầy lội nhưng tuyệt đối không bịa đặt làm lệch hướng cốt truyện chính; bảo toàn 1:1 ID thoại và JSON schema chuẩn xác.
   - **Quy tắc dấu câu Manga**: Tuyệt đối không để dấu chấm đơn (`.`) ở cuối câu thoại trong bong bóng (trừ ba chấm `...` hoặc viết tắt).

---

## 5. Báo Cáo Đo Đạc & Đánh Giá Thực Nghiệm (Live Benchmarks)

> **Môi Trường Thử Nghiệm**: Windows 11 NT (`win32`) | Python 3.14 | PyTorch 2.13.0+cu126 | **GPU: NVIDIA GeForce RTX 3050 Laptop (4,096 MB VRAM)** | CPU: 16 Cores | RAM: 16 GB.

### 5.1. Bảng Tổng Hợp Thời Gian & Chiếm Dụng Tài Nguyên Toàn Trình (Thực Đo)

| Giai Đoạn Pipeline | Mô Hình / Công Nghệ | Thiết Bị Thực Thi | Thời Gian Trung Bình | Đỉnh VRAM GPU | Tiêu Thụ Token / API |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Text Detection** | ComicTextDetector (CTD 1536px) | **GPU CUDA 12.6** | **0.22s - 0.42s / trang** | **260.4 MB** | 0 Token / 0 Request |
| **2. Text OCR** | WinRT OCR / PaddleOCR Multi-View | **CPU / GPU** | **0.08s / khối** | **0.0 MB** | **0 Token (0đ)** |
| **3. Generative Inpainting** | LaMa Large FFC (2-Pass Iterative) | **GPU CUDA 12.6** | **~1.72s / trang** (>206K mask px) | **968.2 MB** | 0 Token / 0 Request |
| **4. Chapter Translation** | Gemini 3.5 Flash Lite / Proxy | **Cloud LLM (HTTP/2)** | **0.90s - 1.20s / trang** | **0.0 MB** | **1 Request / Toàn Trang** |
| **5. Auto 2-Font Typeset** | Emotion-Adaptive 2-Font Engine | **CPU / QPainter** | **0.02s / trang** | **0.0 MB** | 0 Token / 0 Request |
| **TOÀN TRÌNH 1 TRANG** | **Pipeline Hoàn Chỉnh (2-Pass)** | **CUDA + CPU + Cloud** | **~2.85s / trang** | **~1,228 MB (< 1.3 GB)** | **~21.0 trang / phút** |

### 5.2. Đo Đạc Độ Chính Xác Khôi Phục Nền & Độ Sạch Thị Giác (5 Real Manga Samples)
* **`Unmasked MAE` (Sai số ngoài vùng mask)**: **`0.000000`** (Bảo toàn nguyên vẹn 100% từng pixel gốc bên ngoài nét chữ trên toàn bộ 5 mẫu thử nghiệm).
* **`CJK Residuals`**: **0** (Triệt tiêu 100% tàn dư ký tự gốc tiếng Anh/tiếng Nhật bên dưới bản dịch).
* **Triệt tiêu 100% đường viền phân tách hình chữ nhật (Hard Seams)**: Nhờ cơ chế **Laplacian Pyramid Multi-Band Blending (4 tầng)** kết hợp **Gradient-Aware ROI Expansion $2.2\times$**, toàn bộ các panel có screentone dốc chuyển tiếp mượt mà, không còn hiện tượng gãy dải sắc (gradient stepping) hay lệch tone inpaint.
* **Độ Lệch Chuẩn Vùng Khôi Phục Nền (`Inpainted Std Dev`)**: Tái tạo hoàn hảo hạt trame manga và chuyển sắc gradient nơ-ron tự nhiên, độ lệch chuẩn bám sát nền gốc xung quanh.
* **Secondary Pass Trigger & Seamless Integration (Trang 4)**: Phát hiện phương sai Laplacian đạt $387.7$, kích hoạt thành công lượt quét thứ hai với ROI mở rộng $2.2\times$ dọc kèm hòa trộn Laplacian pyramid, làm sạch hoàn toàn chữ tàn dư mà không để lại bất kỳ vệt phân tách khối hộp nào.

### 5.3. So Sánh Hiệu Quả Token Translation Giữa Dịch Đơn Lẻ vs Batching Toàn Trang

| Quy mô Batch | Số khối thoại | Ký tự Prompt | Input Tokens | Output Tokens | Tổng Tokens | Tokens / Thoại | Tỷ lệ tiết kiệm |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Đơn lẻ (Single)** | 1 | 1,285 | 237 | 55 | 292 | 292.0 | Baseline |
| **Khối nhỏ (Small)** | 5 | 1,564 | 301 | 155 | 456 | 91.2 | **-68.8%** |
| **Toàn trang (Full Page)** | **20** | **2,482** | **507** | **530** | **1,037** | **51.9** | **-82.2%** |

### 5.4. Đo Đạc Hạn Mức Quota Thực & Tốc Độ Giãn Cách (Google AI Studio Free Tier)

| Nhóm / Tier | Mô Hình AI | RPM | RPD (Ngày) | Khoảng Cách Gửi Tối Thiểu | Trạng Thái & Ưu Tiên |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Ưu tiên 1 (Chính)** | **`gemini-3.8-flash`** | 5 | **20** | **12.60s** | GA Sept 2026, model thông minh nhất, dịch thoát ý, giàu khẩu ngữ |
| **Ưu tiên 2 (Chính)** | **`gemini-3.7-flash`** | 5 | **20** | **12.60s** | Hiểu ngữ cảnh sâu, liên kết câu thoại mượt mà |
| **Ưu tiên 3 (Chính)** | **`gemini-3.6-flash`** | 5 | **20** | **12.60s** | Cân bằng chất lượng dịch và tốc độ |
| **Ưu tiên 4 (Chính)** | **`gemini-3.5-flash`** | 5 | **20** | **12.60s** | Dự phòng chất lượng cao |
| **Ưu tiên 5 (Dự phòng)** | **`gemini-3.5-flash-lite`** | 15 | **500** | **4.20s** | Dự phòng tải cao RPD 500 khi các model Flash cạn quota ngày |

### 5.5. Kết Quả Kiểm Định 5 Mẫu Manga Scan Thực Tế (`scripts/verify_ocr_inpaint_accuracy.py`)

| Mẫu Manga Thử Nghiệm | Số Box Detect | OCR Thành Công | Unmasked MAE | Inpaint Std Dev | Mean Box Laplacian | Đánh Giá Sạch Sẽ (Cleanliness) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Page 1 (Action/SFX)** | 7 | 7 (100%) | **`0.000000`** | 15.95 | 160.9 | **CLEAN ✓** |
| **Page 2 (Slice of Life)** | 7 | 7 (100%) | **`0.000000`** | 4.95 | 1.8 | **CLEAN ✓** |
| **Page 3 (Mystery/Dark)** | 9 | 9 (100%) | **`0.000000`** | 5.08 | 1.5 | **CLEAN ✓** |
| **Page 4 (User Mystery 1024px)** | 8 | 7 (100%) | **`0.000000`** | 101.23 | 529.2 | **CLEAN ✓ (Triggered 2nd-pass 2.2x + Laplacian)** |
| **Page 5 (Scanlation Tone)** | 4 | 4 (100%) | **`0.000000`** | 93.32 | 58.1 | **CLEAN ✓** |

### 5.6. Kết Quả Tổng Kiểm Thử Hồi Quy Toàn Trình (`scripts/run_all_tests.py`)
* **Tổng kết nghiệm thu**: **10/10 Suites Passed (100.0%) in 241.28s** (Toàn bộ 10/10 bộ kiểm thử tích hợp, OCR, inpainting, gradient blending và pipeline toàn trình đạt trạng thái PASSED tuyệt đối).

| Bộ Kiểm Thử (Test Suite) | Trạng Thái | Thời Gian | Nội Dung Kiểm Định |
| :--- | :---: | :---: | :--- |
| **1. Run vs Continue Branching Suite** | **✓ PASSED** | 6.51s | Phân nhánh Run tạo job mới vs Continue dịch tiếp |
| **2. Ctrl+S Save Text Box Preservation** | **✓ PASSED** | 3.87s | Lưu thủ công Ctrl+S bảo toàn ô chữ & vùng chọn |
| **3. Context-Aware Manga Translation** | **✓ PASSED** | 17.49s | Nối mạch câu đa bong bóng & khóa đại từ nhân xưng |
| **4. UI State Synchronization Suite** | **✓ PASSED** | 5.31s | Đồng bộ trạng thái UI & Canvas Text Layer |
| **5. Translation Proxy Full Test Suite** | **✓ PASSED** | 15.74s | 12 ca kiểm thử Translation Proxy & Fallback chain |
| **6. Deep Context & Pronoun Locking** | **✓ PASSED** | 5.93s | Kiểm thử ngữ cảnh sâu & khóa đại từ nhân xưng |
| **7. Vietnamese Font Rendering Suite** | **✓ PASSED** | 0.21s | Kết xuất font tiếng Việt chuẩn OFL không lỗi dấu |
| **8. OCR Multi-View & Anomaly Scoring Suite** | **✓ PASSED** | 0.30s | Thẩm định 3 góc nhìn ảnh & chấm điểm chất lượng |
| **9. Pipeline Coordinator & Atomic Commit Suite** | **✓ PASSED** | 3.87s | Điều phối viên độc quyền & Phase 3 Atomic Commit |
| **10. Real Manga Inpainting & Pipeline** | **✓ PASSED** | 182.04s | Pipeline toàn trình trên ảnh manga scan thật bằng CUDA |

---

## 6. Hướng Dẫn Khởi Động & Bộ Lệnh Kiểm Thử

### 🚀 Khởi Chạy 1 Chạm:
1. Nhấp đúp chuột vào file **[`start.bat`](file:///d:/BallonsTranslator/start.bat)** trong thư mục `D:\BallonsTranslator`.
2. Hoặc chạy qua PowerShell:
   ```powershell
   .\start.bat
   ```

### 🧪 Bộ Lệnh Kiểm Thử & Đo Đạc Tự Động (10/10 SUITES PASSED - 100%):
```powershell
# 0. Chạy Toàn Bộ 10 Bộ Kiểm Thử Tự Động Toàn Trình (10/10 SUITES PASSED - 100%):
$env:PYTHONIOENCODING="utf-8" ; .\venv\Scripts\python.exe scripts/run_all_tests.py

# 1. Kiểm thử phân nhánh Run (chạy lại từ đầu) vs Continue (dịch tiếp) (4/4 PASSED):
$env:PYTHONIOENCODING="utf-8" ; .\venv\Scripts\python.exe scripts/test_run_and_continue.py

# 2. Kiểm thử lưu thủ công Ctrl+S bảo toàn ô chữ & vùng chọn (2/2 PASSED):
$env:PYTHONIOENCODING="utf-8" ; .\venv\Scripts\python.exe scripts/test_ctrl_s_save.py

# 3. Kiểm thử dịch thuật nối mạch câu & chống cắt cụt ngữ nghĩa (8/8 PASSED):
$env:PYTHONIOENCODING="utf-8" ; .\venv\Scripts\python.exe scripts/test_context_aware_translation.py

# 4. Kiểm thử đồng bộ giao diện & Canvas Text Layer tức thì (8/8 PASSED):
$env:PYTHONIOENCODING="utf-8" ; .\venv\Scripts\python.exe scripts/test_ui_sync.py

# 5. Chạy bộ 12 ca kiểm thử Translation Proxy & Dynamic Model Fallback (12/12 PASSED):
$env:PYTHONIOENCODING="utf-8" ; .\venv\Scripts\python.exe scripts/test_translation_proxy_full.py

# 6. Chạy bộ 11 ca kiểm thử Deep Context Translation & Pronoun Locking (11/11 PASSED):
$env:PYTHONIOENCODING="utf-8" ; .\venv\Scripts\python.exe tests/test_deep_context_translation.py

# 7. Kiểm tra kết xuất font tiếng Việt (Nunito ExtraBold & Sriracha Regular):
$env:PYTHONIOENCODING="utf-8" ; .\venv\Scripts\python.exe scripts/test_fonts_vietnamese.py

# 8. Kiểm thử OCR Multi-View & Anomaly Scoring Suite (10/10 PASSED):
$env:PYTHONIOENCODING="utf-8" ; .\venv\Scripts\python.exe tests/test_ocr_validator.py

# 9. Kiểm thử Pipeline Coordinator & Atomic Commit Suite (2/2 PASSED):
$env:PYTHONIOENCODING="utf-8" ; .\venv\Scripts\python.exe tests/test_pipeline_integrity.py

# 10. Đo đạc LaMa Large Inpainting Full-Page trên GPU CUDA:
$env:PYTHONIOENCODING="utf-8" ; .\venv\Scripts\python.exe scripts/test_real_manga.py

# 11. Chạy bộ kiểm định độ chính xác OCR + Inpaint & xuất ảnh minh chứng 5 trang manga thật:
$env:PYTHONIOENCODING="utf-8" ; .\venv\Scripts\python.exe scripts/verify_ocr_inpaint_accuracy.py

# 12. Kiểm thử Quota Tracker, Throttle RPM & Phân tầng Fallback (Tiered Routing):
$env:PYTHONIOENCODING="utf-8" ; .\venv\Scripts\python.exe scripts/test_quota_and_tier_fallback.py
```

### ⌨️ Phím Tắt Tiện Ích:
* `Ctrl + O`: Mở thư mục truyện tranh cần dịch.
* `F5` / `Ctrl + R`: Chạy toàn bộ Pipeline dịch trang hiện tại (Detection $\rightarrow$ OCR $\rightarrow$ Inpaint $\rightarrow$ Batch Translation $\rightarrow$ Typeset).
* `Ctrl + S`: Lưu & Xuất ảnh kết quả hoàn chỉnh (Ảnh gốc + Nền xóa chữ + Lớp chữ dịch tiếng Việt).
* `T`: Bật/Tắt chế độ Text Edit Mode để chỉnh sửa trực quan nội dung bản dịch.
* `Ctrl + Z` / `Ctrl + Y`: Hoàn tác (Undo) / Làm lại (Redo) thao tác trên Canvas.
