import torchvision
import torch

def get_maskrcnn(num_classes=2, hidden_layer=256):
    """Get a Mask R-CNN model with a custom number of classes and hidden layer size.

    Args:
        num_classes (int, optional): Number of classes (including background). Defaults to 2.
        hidden_layer (int, optional): Number of hidden units in the mask predictor. Defaults to 256.

    Returns:
        torchvision.models.detection.MaskRCNN: The modified Mask R-CNN model.
    """
    
    # load an instance of Mask R-CNN with a ResNet50 FPN backbone
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='COCO_V1')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the box predictor with a new one
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # replace mask predictor with a new one
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    model.roi_heads.mask_predictor = mask_predictor
    model.to(torch.device("cuda"))
    
    return model
