import sys, os
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logging
setup_logging('logs')
from modules import init_module_registries, TRANSLATORS
init_module_registries()

from utils.gemini_proxy_launcher import ensure_gemini_proxy_running
ensure_gemini_proxy_running()

models_to_test = [
    'gemini-3.7-flash',
    'gemini-3.6-flash',
    'gemini-3.5-flash-lite',
]

test_inputs = [
    "おい、待てよ！放課後どこへ行く気だ？",
    "寒くない？私のマフラー、使っていいよ。",
    "貴様が魔王軍の幹部か…覚悟しろ！",
]

print("=" * 60)
print("  TESTING 3 STREAMLINED GEMINI MODELS (3.7, 3.6, 3.5-Lite)")
print("=" * 60)

LLM_Cls = TRANSLATORS.module_dict['LLM_API_Translator']

for model_name in models_to_test:
    print(f"\n--- Testing Model: {model_name} ---")
    translator = LLM_Cls(
        '日本語', 'Tiếng Việt',
        provider='Gemini Proxy',
        model=model_name,
        endpoint='http://127.0.0.1:8080/v1',
        raise_unsupported_lang=False
    )
    print(f"Initialized {translator.name} with model: {translator.model}")
    print("Testing translation setup and execution...")
    try:
        results = translator._translate(test_inputs)
        print("Results received:")
        for orig, trans in zip(test_inputs, results):
            print(f"  JP: {orig}")
            print(f"  VI: {trans}")
        print(f"Model {model_name}: SUCCESS!")
    except Exception as e:
        print(f"Model {model_name} result: {e}")

print("\n" + "=" * 60)
print("  ALL 3 GEMINI MODELS TEST COMPLETED!")
print("=" * 60)
