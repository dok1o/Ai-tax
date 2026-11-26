# Round 3 integration handoff

- ARENA fixes through `8548911` cover router thresholds, correctness decoding and fixture input selection. T-016 is outside this review.
- T-014 supplies the trilingual dataset and saved token counts. All 180 prompt rows await native verification.
- T-015 adds the Ollama and Gemini collectors. Offline collection uses the fixture client. Live collection requires model selection and human authorization.
- Export a worksheet for the imported questions before native review. RELEASE.md records the outstanding offline tokenizer and release checks.
- Resume on dev with `python run.py check`.
