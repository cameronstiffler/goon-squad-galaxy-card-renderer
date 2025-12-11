# Cardrender Quick Commands

Run all commands from the repository root.

## Render decks
- PCU deck grid: `python3 card_generator.py -pcu -grid`
- PCU deck grid with duplicates respected: `python3 card_generator.py -pcu -grid -dup`
- Swap `-pcu` for `-narc` or `-meat` to target those decks. Use `-auto` to generate missing art; add `-fix` to backfill missing JSON fields.

## Switching AI providers
- `.env` is auto-loaded when the script starts. Leave the OpenAI lines uncommented (and Gemini commented) to use OpenAI defaults.
- To use Gemini instead, comment the OpenAI lines, uncomment the Gemini block, and set your key/models; optionally set `MODEL_PROVIDER` to override when both keys are present. Gemini usage requires the `google-generativeai` Python package.
