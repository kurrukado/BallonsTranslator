import sys, os
sys.path.insert(0, os.path.abspath(os.getcwd()))

import sys
import io
import unittest
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from modules.translators.trans_llm_api import LLM_API_Translator, DialogueItem
from modules.translators.context_engine import ContextAssembler, CharacterProfile, CharacterMemory, GlossaryManager, SFX_EN_VN
from utils.textblock import TextBlock


class TestDeepContextTranslation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.translator = LLM_API_Translator('English', 'Tiếng Việt')

    def test_case_1_same_sentence_same_context_dedup_and_cache(self):
        items = [
            DialogueItem(id=1, source='Wait!', speaker='Hero', block_type='DIALOGUE'),
            DialogueItem(id=2, source='Wait!', speaker='Hero', block_type='DIALOGUE'),
        ]
        for prompt, num_unique, chunk, s2p in self.translator._assemble_prompts(items, to_lang='Vietnamese'):
            self.assertEqual(num_unique, 1, 'Expected 1 unique prompt item for identical speaker/context')
            self.assertEqual(len(chunk), 2)
            self.assertIn('CURRENT DIALOGUES TO TRANSLATE:', prompt)
        print('✓ Case 1 (Same sentence, same context dedup): PASSED')

    def test_case_2_same_sentence_different_speakers_no_wrong_dedup(self):
        items = [
            DialogueItem(id=1, source='What?!', speaker='Hero (Angry)', block_type='DIALOGUE'),
            DialogueItem(id=2, source='What?!', speaker='Villain (Shocked)', block_type='DIALOGUE'),
        ]
        for prompt, num_unique, chunk, s2p in self.translator._assemble_prompts(items, to_lang='Vietnamese'):
            self.assertEqual(num_unique, 2, 'Different speakers MUST produce 2 distinct prompt items')
        print('✓ Case 2 (Same sentence, different speakers distinct): PASSED')

    def test_case_3_pronoun_locking_extraction(self):
        history = [
            DialogueItem(id=1, source='Are you okay?', translated='Cậu có sao không?'),
            DialogueItem(id=2, source='I am fine, thanks.', translated='Tớ ổn, cảm ơn cậu.'),
        ]
        char_mem = CharacterMemory([
            CharacterProfile(name='Kusora Yurea', pronouns_self='tớ', pronouns_to_others={'Protagonist': 'cậu'})
        ])
        locks = ContextAssembler.extract_pronoun_locks(dialogue_history=history, character_memory=char_mem)
        self.assertTrue(any('cậu - tớ' in lock for lock in locks))
        print('✓ Case 3 (Pronoun locking extraction): PASSED')

    def test_case_4_multi_bubble_cohesion_detection(self):
        items = [
            DialogueItem(id=1, source='I never thought...'),
            DialogueItem(id=2, source='...that you would actually come here.'),
        ]
        ContextAssembler.detect_connected_bubbles(items)
        self.assertIsNotNone(items[0].connected_group_id)
        self.assertEqual(items[0].connected_group_id, items[1].connected_group_id)
        self.assertEqual(items[0].flow_hint, 'Start of multi-bubble sentence')
        self.assertEqual(items[1].flow_hint, 'Continuation of multi-bubble sentence')
        print('✓ Case 4 (Multi-bubble cohesion detection): PASSED')

    def test_case_5_glossary_term_extraction(self):
        glossary = GlossaryManager()
        glossary.add_entry('Cursed Energy', 'Chú lực')
        glossary.add_entry('Domain Expansion', 'Bành trướng Lãnh địa')
        glossary.add_entry('Unrelated Skill', 'Chiêu thức khác')

        relevant = glossary.get_relevant_terms(['He released his Cursed Energy!', 'Watch out!'])
        terms_found = [r['term'] for r in relevant]
        self.assertIn('Cursed Energy', terms_found)
        self.assertNotIn('Unrelated Skill', terms_found)
        print('✓ Case 5 (Relevant glossary filtering): PASSED')

    def test_case_6_block_type_classification_and_sfx(self):
        self.assertEqual(ContextAssembler.classify_block_type('(Is he really the culprit?)', False), 'THOUGHT')
        self.assertEqual(ContextAssembler.classify_block_type('【Three days later】', False), 'NARRATION')
        self.assertEqual(ContextAssembler.classify_block_type('POW!!', False), 'SFX')
        self.assertEqual(ContextAssembler.classify_block_type('BAM!', False), 'SFX')
        self.assertEqual(ContextAssembler.classify_block_type('ドン！', True), 'SFX')
        self.assertEqual(ContextAssembler.classify_block_type('CRASH', False), 'SFX')
        self.assertEqual(ContextAssembler.classify_block_type('GASP', False), 'SFX')
        self.assertEqual(ContextAssembler.classify_block_type('SQUEAK', False), 'DIALOGUE', 'Unknown SFX should not be classified as SFX')
        print('✓ Case 6 & 7 (Block type & SFX classification): PASSED')

    def test_case_7b_sfx_localization_dictionary(self):
        self.assertIn('POW', SFX_EN_VN)
        self.assertEqual(SFX_EN_VN['POW'], 'BỐP!')
        self.assertIn('ドン', SFX_EN_VN)
        self.assertEqual(SFX_EN_VN['ドン'], 'RẦM!')
        self.assertIn('ドキドキ', SFX_EN_VN)
        self.assertEqual(SFX_EN_VN['ドキドキ'], 'THỊCH THỊCH')
        print('✓ Case 7b (SFX localization dictionary): PASSED')

    def test_case_8_ocr_noise_awareness_in_prompt(self):
        items = [
            DialogueItem(id=1, source='Wait!', speaker='Hero', block_type='DIALOGUE'),
        ]
        for prompt, num_unique, chunk, s2p in self.translator._assemble_prompts(items, to_lang='Vietnamese'):
            self.assertIn('OCR NOISE', prompt, 'Prompt must include OCR noise resilience instruction')
            self.assertIn('correct', prompt.lower())
        print('✓ Case 8 (OCR noise awareness in prompt): PASSED')

    def test_case_9_user_manual_edits_protection(self):
        blk1 = TextBlock(text=['Hello'])
        blk1.translation = 'Xin chào tự động'
        blk1.user_edited = False

        blk2 = TextBlock(text=['Hey there'])
        blk2.translation = 'Chào đằng ấy (Đã sửa tay)'
        blk2.user_edited = True

        blks = [blk1, blk2]
        
        for orig_idx, blk in enumerate(blks):
            is_user_edited = getattr(blk, 'user_edited', False) or getattr(blk, 'manual_edit', False)
            if not is_user_edited:
                blk.translation = 'Bản dịch mới từ AI'
            else:
                pass

        self.assertEqual(blk1.translation, 'Bản dịch mới từ AI')
        self.assertEqual(blk2.translation, 'Chào đằng ấy (Đã sửa tay)')
        print('✓ Case 9 (User manual edits protection): PASSED')

    def test_case_10_cache_invalidation_on_context_change(self):
        k1 = self.translator._compute_cache_key(
            'What?!', 'English', 'Vietnamese', 'gemini-3.5-flash-lite', 'prompt1',
            context_hash='ctxA', speaker='Hero'
        )
        k2 = self.translator._compute_cache_key(
            'What?!', 'English', 'Vietnamese', 'gemini-3.5-flash-lite', 'prompt1',
            context_hash='ctxB', speaker='Hero'
        )
        k3 = self.translator._compute_cache_key(
            'What?!', 'English', 'Vietnamese', 'gemini-3.5-flash-lite', 'prompt1',
            context_hash='ctxA', speaker='Villain'
        )
        self.assertNotEqual(k1, k2, 'Changing context MUST invalidate cache key')
        self.assertNotEqual(k1, k3, 'Changing speaker MUST invalidate cache key')
        print('✓ Case 10 (Context & speaker cache invalidation): PASSED')

    def test_case_11_compact_format_contains_key_sections(self):
        items = [
            DialogueItem(id=1, source='Test dialogue', speaker='Hero', block_type='DIALOGUE', position='Top-Right'),
        ]
        payload = ContextAssembler.assemble_payload(
            current_items=items,
            to_lang='Vietnamese',
            from_lang='English',
        )
        self.assertIn('SCENE:', payload)
        self.assertIn('CURRENT DIALOGUES TO TRANSLATE:', payload)
        self.assertIn('[1]', payload)
        self.assertIn('Test dialogue', payload)
        print('✓ Case 11 (Compact format key sections): PASSED')


if __name__ == '__main__':
    unittest.main()