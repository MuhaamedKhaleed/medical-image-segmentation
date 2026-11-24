# Measured no-cost CPU quality experiment

64officialtraining/16development/16firstofficialtest fields,40epochs,width8,lr.001. Checkpoint selected on development, then testDice.89676. Prior development.075 used different subset/config, not controlled test improvement. Microscopy notMRI/clinical. PeakRSS613MiB,training25sec.

Run only explicitly, with previously prepared local permitted data and a new output. No automatic downloads or import-time execution. Exact measured result is quality-output.json. This is a reproducible development experiment, not a productionquality guarantee.

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/quality_cpu.py --data LOCAL_PREPARED_SOURCE --output NEW_OUTPUT
```

Test sets described as finalized have now been observed. Future choices must not be justified with their scores as if they remained unseen. Retain source terms/attribution from real-data.md. No model or raw text/photos committed.
