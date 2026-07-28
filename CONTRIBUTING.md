# Contributing

Thank you for helping improve ADCT.

1. Create a focused branch.
2. Install `pip install -e ".[dev]"`.
3. Add tests for behavioral changes.
4. Run `pytest` and `ruff check src tests examples`.
5. Keep robot-, dataset-, and checkpoint-specific paths out of source files.
6. Do not commit private data, credentials, large checkpoints, or generated
   roll-out videos.

For experiment reports, include the configuration, dataset split, random
seed, checkpoint hash, number of roll-outs, and success criterion.

