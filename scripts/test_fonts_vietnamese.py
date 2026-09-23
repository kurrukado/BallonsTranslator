import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent

font_baloo_path = str(PROJECT_ROOT / "fonts" / "Yuki-CCMarianChurchlandJournal.ttf") if (PROJECT_ROOT / "fonts" / "Yuki-CCMarianChurchlandJournal.ttf").exists() else "C:/Windows/Fonts/arial.ttf"
font_sriracha_path = str(PROJECT_ROOT / "fonts" / "Yuki-Ripsnort BB.ttf") if (PROJECT_ROOT / "fonts" / "Yuki-Ripsnort BB.ttf").exists() else "C:/Windows/Fonts/arial.ttf"

out_dir = PROJECT_ROOT / "tmp" / "live_output"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "vietnamese_font_validation.png"

img = Image.new('RGB', (900, 520), color=(255, 255, 255))
draw = ImageDraw.Draw(img)

font_dialogue_title = ImageFont.truetype(font_baloo_path, size=24)
font_dialogue = ImageFont.truetype(font_baloo_path, size=22)

font_sfx_title = ImageFont.truetype(font_sriracha_path, size=24)
font_sfx = ImageFont.truetype(font_sriracha_path, size=22)

# 1. Dialogue Font (Baloo 2 Bold)
draw.rectangle([(20, 20), (880, 240)], outline=(200, 200, 200), width=2)
draw.text((40, 35), "1. DIALOGUE PRESET (Font: Baloo 2 Bold)", fill=(180, 20, 20), font=font_dialogue_title)
draw.text((40, 75), "Hội thoại: \"Này cậu, ngày mai có bài kiểm tra toán đấy, đã ôn bài chưa?\"", fill=(0, 0, 0), font=font_dialogue)
draw.text((40, 115), "Dấu tiếng Việt đầy đủ: à á ả ã ạ / è é ẻ ẽ ẹ / ì í ỉ ĩ ị / ò ó ỏ õ ọ / ù ú ủ ũ ụ / ỳ ý ỷ ỹ ỵ", fill=(30, 80, 180), font=font_dialogue)
draw.text((40, 155), "Dấu nguyên âm phức tạp: ầ ấ ẩ ẫ ậ / ừ ứ ử ữ ự / ồ ố ổ ỗ ộ / ằ ắ ẳ ẵ ặ / ê ế ề ể ễ ệ / đ", fill=(30, 80, 180), font=font_dialogue)
draw.text((40, 195), "Khẳng định: \"Tớ tin chắc chắn chúng ta sẽ cùng nhau vượt qua mọi thử thách!\"", fill=(0, 100, 40), font=font_dialogue)

# 2. Narration/SFX Font (Sriracha)
draw.rectangle([(20, 260), (880, 490)], outline=(200, 200, 200), width=2)
draw.text((40, 275), "2. NARRATION / SFX PRESET (Font: Sriracha)", fill=(180, 20, 20), font=font_sfx_title)
draw.text((40, 315), "Tiếng động (SFX): \"ĐÙNG! XOẸT! RẦM RẦM! THỊCH THỊCH! VÚT! BỐP!\"", fill=(0, 0, 0), font=font_sfx)
draw.text((40, 355), "Lời dẫn truyện: \"Mùa hè năm ấy, những tán phượng vĩ rực lửa bên góc sân trường xưa...\"", fill=(30, 80, 180), font=font_sfx)
draw.text((40, 395), "Kiểm tra dấu ngã, hỏi, nặng: ẵ, ữ, ộ, ử, ễ, ặ, ỳ, đ, ỷ, ự, ỡ, ở, ớ, ờ", fill=(0, 100, 40), font=font_sfx)
draw.text((40, 435), "Cảm thán ngoài khung: \"(Bầu không khí bỗng trở nên căng thẳng tột độ...)\"", fill=(100, 50, 120), font=font_sfx)

img.save(str(out_file))
print(f"[SUCCESS] Font verification image generated at: {out_file}")
