# Agent Learnings & Architectural Knowledge Base

## Lesson 1: Direct translation must bypass batch buffer
- **Problem**: Manual editing and re-translating single speech bubbles was delayed by chapter batching.
- **Root Cause**: UI edits were erroneously routed through the same aggregation state buffer as RUN mode.
- **Fix**: Introduced explicit `TranslationMode.DIRECT` with `translate_single` returning plain string without queuing or Pydantic JSON wrapping.
- **Regression Test**: `test_08_direct_mode_buffer_bypass` and `test_09_concurrent_direct_during_run_mode` in `scripts/test_translation_proxy_full.py`.

## Lesson 2: Page Completion Barrier for Clean Inpainted Backgrounds
- **Problem**: Translating before Inpaint or before OCR completion caused race conditions and typesetting artifacts on dirty backgrounds.
- **Root Cause**: Pipeline fired translation per-page before downstream inpainting finished.
- **Fix**: In `ImgtransThread._imgtrans_pipeline`, Detection $\to$ OCR $\to$ Inpaint execute first for all pages. Text is queued to `TranslationProxy` only after OCR and Inpainting succeed, followed by a single consolidated Chapter Batch dispatch.
- **Regression Test**: `test_01_async_page_arrival_and_batch_queuing` and `test_chapter_batch_pipeline_flow`.

## Lesson 3: 1:1 ID Validation & Set Equality Check
- **Problem**: LLMs occasionally omit IDs, hallucinate extra IDs, or scramble the sequence order.
- **Root Cause**: Naive JSON decoding without strict set validation allowed mismatched text assignments.
- **Fix**: Implemented `validate_and_unpack` in `TranslationProxy` verifying `set(input_ids) == set(output_ids)` and re-indexing by `(page_index, reading_order, id)`.
- **Regression Test**: `test_02_structured_json_id_preservation`, `test_03_shuffled_id_restoration`, `test_04_omitted_id_validation_failure`, `test_05_unknown_id_rejection`.

## Lesson 4: Dynamic Model Fallback Chain on 429 / Quotas
- **Problem**: Rate limits (HTTP 429) or timeouts on primary model (`gemini-3.5-flash-lite`) blocked the pipeline.
- **Root Cause**: Fixed single-model dispatch without automated fallback sequence.
- **Fix**: Implemented fallback chain: `gemini-3.5-flash-lite` $\to$ `gemini-3.5-flash` $\to$ `gemini-3.6-flash` $\to$ `gemini-3.7-flash` $\to$ `gemini-2.5-flash-lite` with exponential backoff and jitter.
- **Regression Test**: `test_06_http_429_rate_limit_fallback` and `test_07_primary_model_timeout_fallback`.

## Lesson 5: UI State Synchronization & Checkbox Toggle Safety
- **Problem**: Canvas text layer was hidden after complete Run translation, requiring the user to click buttons or restart the app.
- **Root Cause**: Calling `.click()` on `QCheckBox` inverted its boolean state instead of asserting it, and `saveCurrentPage`'s `render_result_img` temporarily hid `textLayer` without restoring it.
- **Fix**: Replaced `.click()` with `.setChecked(True/False)` in `MainWindow.saveCurrentPage`, called `self.st_manager.updateSceneTextitems()`, `self.canvas.updateCanvas()`, and explicit `self.canvas.textLayer.show()` in `on_pagtrans_finished` and `on_imgtrans_pipeline_finished`.
- **Regression Test**: `scripts/test_ui_sync.py` (8/8 tests passed).

## Lesson 6: Context-Aware Manga Translation & Multi-Bubble Sentence Stitching
- **Problem**: Translating OCR blocks as isolated sentences caused severe semantic truncation (e.g. `their cohabitation life starts a new phase` translated as bare noun phrase `Cuộc sống chung`, dropping `their`, `starts`, and `a new phase`), or broke multi-bubble sentences into disjointed fragments.
- **Root Cause**: OCR bounding boxes represent visual segments, not linguistic clauses. The prompt lacked explicit instructions on visual segment stitching, anti-truncation preservation, and title vs clause distinction.
- **Fix**: Refactored `system_instruction` in `trans_llm_api.py` and `module_manager.py` with multi-bubble sentence stitching, strict anti-truncation preservation of subjects/predicates/modifiers, standalone title vs complete clause handling, and compact context hints for Direct Mode.
- **Regression Test**: `scripts/test_context_aware_translation.py` (8/8 tests passed).

## Lesson 7: Manual Ctrl+S Save Active Text Box & Selection Preservation
- **Problem**: Pressing `Ctrl + S` caused active translated text boxes and selection bounding frames (`txtblkShapeControl`) to disappear from the Canvas, and could wipe out translations if the text item document was being synchronized.
- **Root Cause**: `render_result_img()` during save called `self.clearSelection()` to prevent selection handles from leaking into the exported image, which detached `txtblkShapeControl.blk_item` to `None`. In addition, `updateTextBlkList` wiped `blk.translation` if the item document was empty instead of checking `trans_pair.e_trans`.
- **Fix**: Saved `active_blkitem = self.st_manager.txtblkShapeControl.blk_item` and editing state before render, explicitly restored selection and shape control binding after render, ensured `self.canvas.textLayer.show()`, and safeguarded `updateTextBlkList` to sync from `trans_pair.e_trans` before resetting.
- **Regression Test**: `scripts/test_ctrl_s_save.py` (2/2 tests passed).

