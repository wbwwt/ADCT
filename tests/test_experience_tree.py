from adct.config import ExperienceTreeConfig, TargetRegionConfig
from adct.experience_tree import ExperienceState, ExperienceTree
from adct.types import Detection


def detection(label: int, center_x: float, center_y: float, score: float = 0.9) -> Detection:
    return Detection(
        label=label,
        box=(center_x - 2, center_y - 2, center_x + 2, center_y + 2),
        score=score,
    )


def fitted_tree() -> ExperienceTree:
    config = ExperienceTreeConfig(
        target_region=TargetRegionConfig(
            center_x=100,
            center_y=100,
            tolerance_x=10,
            tolerance_y=10,
        ),
        min_confidence=0.3,
    )
    episode = [
        [detection(0, 20, 20), detection(1, 30, 30)],
        [detection(0, 100, 100), detection(1, 30, 30)],
        [detection(0, 100, 100), detection(1, 100, 100)],
    ]
    return ExperienceTree(config).fit([episode, episode])


def test_tree_learns_first_entry_order() -> None:
    tree = fitted_tree()
    assert tree.sequence_counts[(0, 1)] == 2
    assert list(tree.root.children) == [0]
    assert list(tree.root.children[0].children) == [1]


def test_tree_selects_expected_unplaced_object() -> None:
    tree = fitted_tree()
    state = ExperienceState()

    selected = tree.select_target(
        [detection(0, 20, 20, 0.8), detection(1, 30, 30, 0.99)],
        state,
    )
    assert selected is not None
    assert selected.label == 0

    changed = tree.update_completed([detection(0, 100, 100)], state)
    assert changed == [0]
    selected = tree.select_target([detection(1, 30, 30)], state)
    assert selected is not None
    assert selected.label == 1


def test_tree_json_round_trip(tmp_path) -> None:
    path = tmp_path / "tree.json"
    fitted_tree().save(path)
    loaded = ExperienceTree.load(path)
    assert loaded.sequence_counts[(0, 1)] == 2
    assert loaded.target_region.center_x == 100

