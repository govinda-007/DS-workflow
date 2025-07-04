from torchmetrics.classification import JaccardIndex
import torch
from tqdm import tqdm

def compute_mean_iou(model, loader, device='cuda'):
    """Compute Mean IoU over a data loader.
    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (str): Device to run the model on ('cuda' or 'cpu').
    Returns:
        float: Mean IoU across all images in the loader."""
    
    model.eval()
    iou_metric = JaccardIndex(task='binary').to(device)
    total_iou = 0
    count = 0
    loader_tqdm = tqdm(loader, leave=True, position=0)
    
    # Note: when doing inference in PyTorch, you need both model.eval() - to deactivate dropout and batch normalization layers - 
    # and torch.no_grad() to deactivate the gradient graph.
    with torch.no_grad():
        # Iterate over the batches in DataLoader
        for imgs, targets in loader_tqdm:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            preds = model(imgs)

            # Squeeze the '1' axis from batching and only store the mask predictions
            pred_masks = [pred['masks'].squeeze(1) for pred in preds]
            true_masks = [t['masks'].squeeze(0) for t in targets]

            # Iterate over the batch
            for pred_mask, true_mask in zip(pred_masks, true_masks):
                if pred_mask.size(0) == 0:
                    # No predictions
                    iou = torch.tensor(0.0, device=device)
                else:
                    # Combine multiple predicted masks into a single mask (the model may predict multiple masks per image)
                    # and binarize
                    pred_binary = (pred_mask > 0.5).max(0).values.int()

                    iou = iou_metric(pred_binary, true_mask) # true_mask is already binary
                
                total_iou += iou.item()
                count += 1

    return total_iou / count