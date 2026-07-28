from test_model import small_config

from adct.checkpoint import load_checkpoint, save_checkpoint
from adct.model import ADCT
from adct.normalization import NormalizationStats


def test_checkpoint_round_trip_preserves_normalization(tmp_path) -> None:
    model = ADCT(small_config())
    stats = NormalizationStats(
        state_mean=(0.0,) * 6,
        state_std=(1.0,) * 6,
        action_mean=(2.0,) * 6,
        action_std=(3.0,) * 6,
    )

    save_checkpoint(model, tmp_path, normalization_stats=stats)
    restored = load_checkpoint(tmp_path)

    assert restored.normalization_stats == stats
