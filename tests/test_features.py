import torch

from adct.features import DetectionFeatureEncoder, normalize_boxes_xyxy


def test_box_normalization() -> None:
    boxes = torch.tensor([[[0.0, 0.0, 640.0, 480.0]]])
    normalized = normalize_boxes_xyxy(boxes, image_width=640, image_height=480)
    assert torch.allclose(normalized, torch.tensor([[[-1.0, -1.0, 1.0, 1.0]]]))


def test_balanced_detection_encoder_shape() -> None:
    encoder = DetectionFeatureEncoder(num_classes=3, dim_model=32)
    labels = torch.tensor([[0, 2], [1, 0]])
    boxes = torch.zeros(2, 2, 4)
    encoded = encoder(labels, boxes)
    assert encoded.shape == (2, 2, 32)
    assert encoder.label_projection.out_features == encoder.box_projection.out_features

