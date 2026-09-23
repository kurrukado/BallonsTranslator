import os
import sys
import time
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from utils.logger import logger as LOGGER

def is_proxy_alive(url="http://127.0.0.1:8080/api/status", timeout=0.2):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BalloonsTranslator"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False

def get_saved_api_key():
    # 1. Try reading from api-key.txt
    txt_path = Path("d:/gemini-proxy/api-key.txt")
    if txt_path.exists():
        try:
            key = txt_path.read_text(encoding="utf-8").strip()
            if key:
                return key
        except Exception:
            pass

    # 2. Try reading from config.json
    try:
        import json
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            key = data.get("module", {}).get("translator_params", {}).get("LLM_API_Translator", {}).get("apikey", "").strip()
            if key:
                return key
    except Exception:
        pass

    # 3. Try env var
    return os.environ.get("GEMINI_API_KEY", "").strip()


def sync_proxy_api_key(api_key: str):
    if not api_key:
        return
    try:
        import json
        payload = json.dumps({"api_key": api_key, "max_retries": 0, "delay_seconds": 0.2}).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:8080/api/config",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "BalloonsTranslator"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                LOGGER.info("[Gemini Proxy] API Key & zero-delay fast fallback synced successfully with proxy runtime.")
    except Exception as e:
        LOGGER.debug(f"[Gemini Proxy] Failed to sync API key via HTTP: {e}")


def stop_gemini_proxy():
    """Stops running gemini-proxy.exe process on Windows."""
    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/F", "/IM", "gemini-proxy.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.5)
        except Exception:
            pass

def restart_gemini_proxy():
    stop_gemini_proxy()
    return ensure_gemini_proxy_running()

def ensure_gemini_proxy_running():
    api_key = get_saved_api_key()

    if is_proxy_alive():
        LOGGER.info("[Gemini Proxy] Proxy is already running on http://127.0.0.1:8080")
        if api_key:
            sync_proxy_api_key(api_key)
        return True

    # Search for gemini-proxy executable
    possible_paths = [
        Path("d:/gemini-proxy/gemini-proxy.exe"),
        Path(__file__).resolve().parent.parent.parent / "gemini-proxy" / "gemini-proxy.exe",
        Path("gemini-proxy.exe")
    ]

    proxy_exe = None
    for p in possible_paths:
        if p.exists():
            proxy_exe = p
            break

    if not proxy_exe:
        LOGGER.warning("[Gemini Proxy] Could not locate gemini-proxy.exe automatically.")
        return False

    try:
        LOGGER.info(f"[Gemini Proxy] Auto-starting proxy from {proxy_exe}...")
        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NO_WINDOW | getattr(subprocess, 'DETACHED_PROCESS', 0x00000008)

        cmd = [str(proxy_exe)]
        if api_key:
            cmd.append(api_key)

        env = os.environ.copy()
        if api_key:
            env["GEMINI_API_KEY"] = api_key

        subprocess.Popen(
            cmd,
            cwd=str(proxy_exe.parent),
            env=env,
            creationflags=creation_flags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Wait up to 3 seconds for server readiness
        for _ in range(15):
            time.sleep(0.2)
            if is_proxy_alive():
                LOGGER.info("[Gemini Proxy] Successfully auto-started and connected on http://127.0.0.1:8080!")
                if api_key:
                    sync_proxy_api_key(api_key)
                return True

        LOGGER.warning("[Gemini Proxy] Proxy process started, but status check timed out.")
        return False
    except Exception as e:
        LOGGER.error(f"[Gemini Proxy] Failed to auto-start proxy: {e}")
        return False
