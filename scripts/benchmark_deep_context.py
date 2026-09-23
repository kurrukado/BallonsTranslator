import sys, io, time, os
sys.path.insert(0, os.path.abspath(os.getcwd()))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from modules.translators.trans_llm_api import LLM_API_Translator, DialogueItem

t = LLM_API_Translator('English', 'Vietnamese')

history = [
    DialogueItem(id=1, source='Kusora, what did you discover at the scene?', translated='Kusora, cầu đã q0t hiện ra điều gì tại hiện thường?'),
    DialogueItem(id=2, source='The victim was attacked from behind.', translated='Nạn nhân đã bị tấn công từ phía sau.')
]

current_page = [
    DialogueItem(id=1, source="I looked into the suspect's room..."),
    DialogueItem(id=2, source="...and found this bizarre talisman."),
    DialogueItem(id=3, source="POW!!"),
    DialogueItem(id=4, source="(This energy... it is definitely not human.)"),
    DialogueItem(id=5, source="What?!"),
    DialogueItem(id=6, source="What?!"),
]

page_ctx = {'dialogue_history': history}

t0 = time.perfcounter()
results = t.translate_dialogue_items(current_page, page_context=page_ctx)
t_el = time.perfcounter() - t0

print(f'=== TRANSLATION RESULTS (Time: {t_el:.2f}s, Tokens: {t.token_count_last}) ===')
for item, res in zip(current_page, results):
    print(f'[{item.block_type:9s}] [{item.source:45s}] -> [{res:45s}] ({item.emotion_tag})')
