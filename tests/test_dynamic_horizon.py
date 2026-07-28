import torch

from adct.config import DynamicHorizonConfig
from adct.dynamic_horizon import ActionChunkBuffer, execution_horizon


def test_execution_horizon_matches_paper_equation() -> None:
    config = DynamicHorizonConfig(
        chunk_size=50,
        confidence_threshold=0.93,
        scaling_factor=50,
    )
    assert execution_horizon(0.95, config) == 50
    assert execution_horizon(0.90, config) == 45
    assert execution_horizon(0.00, config) == 1


def test_action_buffer_keeps_only_selected_prefix() -> None:
    config = DynamicHorizonConfig(
        chunk_size=4,
        confidence_threshold=0.9,
        scaling_factor=4,
    )
    buffer = ActionChunkBuffer(config)
    actions = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    assert buffer.load(actions, confidence=0.5) == 2
    assert len(buffer) == 2
    assert torch.equal(buffer.pop(), actions[0])
    assert torch.equal(buffer.pop(), actions[1])

