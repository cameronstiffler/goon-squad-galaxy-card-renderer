# Cardrender Quick Commands

Run all commands from the repository root.

## Render decks
- PCU deck grid: `python3 card_generator.py -pcu -grid`
- PCU deck grid with duplicates respected: `python3 card_generator.py -pcu -grid -dup`
- Swap `-pcu` for `-narc` or `-meat` to target those decks. Use `-auto` to generate missing art; add `-fix` to backfill missing JSON fields.
