#!/usr/bin/env python3
"""
Downloads Japanese RAW chapters of "Mijuku na Futari de Gozaimasu ga" from Sen Manga
and extracts them directly into target folder (e.g. E:\\truyen-list\\...\\candich\\Chuong_XXX).
"""

import sys
import os
import io
import json
import zipfile
import argparse
import urllib.request
from pathlib import Path
from typing import List, Dict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DEFAULT_TARGET_DIR = Path(r"E:\truyen-list\Mijuku Na Futari De Gozaimasu Ga (Vợ Tôi Là Gái Nguyên Zin)\candich")
API_URL = "https://raw.senmanga.com/api/manga/mijuku-na-futari-de-gozaimasu-ga"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
}

def fetch_chapter_list() -> List[Dict]:
    print(f"[*] Fetching chapter list from Sen Manga API...")
    req = urllib.request.Request(API_URL, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    chapters = data.get("chapterList", [])
    print(f"[+] Found {len(chapters)} chapters on Sen Manga.")
    return chapters

def download_and_extract_chapter(chapter_info: Dict, target_dir: Path, overwrite: bool = True) -> bool:
    ch_num = chapter_info.get("number")
    dl_url = chapter_info.get("download")
    if not dl_url:
        print(f"[-] Chapter {ch_num}: No download link found. Skipping.")
        return False

    folder_name = f"Chuong_{ch_num}"
    ch_dir = target_dir / folder_name

    # Check if already downloaded
    if ch_dir.exists() and not overwrite:
        existing_imgs = list(ch_dir.glob("*.jpg")) + list(ch_dir.glob("*.png"))
        if existing_imgs:
            print(f"[=] Chapter {ch_num}: Folder exists with {len(existing_imgs)} images. Skipping.")
            return True

    ch_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Chapter {ch_num}: Downloading from Google Drive...")

    try:
        req = urllib.request.Request(dl_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            zip_bytes = resp.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            extracted_count = 0
            for file_info in z.infolist():
                if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    # Extract directly into ch_dir without nested directories
                    filename = Path(file_info.filename).name
                    target_file = ch_dir / filename
                    with z.open(file_info) as source, open(target_file, "wb") as target:
                        target.write(source.read())
                    extracted_count += 1

        print(f"[✓] Chapter {ch_num}: Extracted {extracted_count} raw images to {ch_dir}")
        return True
    except Exception as e:
        print(f"[!] Chapter {ch_num} error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Japanese RAW chapters for BalloonsTranslator")
    parser.add_argument("--start", type=float, default=102, help="Start chapter number (default: 102)")
    parser.add_argument("--end", type=float, default=150, help="End chapter number (default: 150)")
    parser.add_argument("--target", type=str, default=str(DEFAULT_TARGET_DIR), help="Target candich directory")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Overwrite existing files")
    parser.add_argument("--single", type=str, default="", help="Download a single chapter (e.g. 102)")
    args = parser.parse_args()

    target_dir = Path(args.target)
    target_dir.mkdir(parents=True, exist_ok=True)

    chapters = fetch_chapter_list()

    # Filter chapters
    if args.single:
        selected = [c for c in chapters if str(c.get("number")) == str(args.single)]
    else:
        selected = []
        for c in chapters:
            try:
                num = float(c.get("number", -1))
                if args.start <= num <= args.end:
                    selected.append(c)
            except ValueError:
                pass

    # Sort ascending by chapter number
    selected.sort(key=lambda c: float(c.get("number", 0)))
    print(f"[+] Selected {len(selected)} chapters to download (from {args.start} to {args.end}).")

    success = 0
    for ch in selected:
        if download_and_extract_chapter(ch, target_dir, overwrite=args.overwrite):
            success += 1

    print("=" * 60)
    print(f"[🎉] Completed: Successfully downloaded & extracted {success}/{len(selected)} chapters into:")
    print(f"     {target_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
