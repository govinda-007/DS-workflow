import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import BeeDataset, collate_fn
from modelling import get_maskrcnn

# Path to the saved model
model_path = "dsworkflow/models/maskrcnn.pt"

# Load the model
model = get_maskrcnn()
model.load_state_dict(torch.load(model_path))

# Load the data
val_img_dir   = "dsworkflow/data/Bees_val_mini"
val_masks_dir = "dsworkflow/data/Bees_masks_val_mini"
val_dataset   = BeeDataset(val_img_dir, val_masks_dir)
val_loader    = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True, collate_fn=collate_fn, num_workers=4)

# Model expects a list of images
model.eval()
with torch.no_grad():
    for imgs, targets in val_loader:
        imgs = [img.to('cuda') for img in imgs]
        targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

        preds = model(imgs)

        # Squeeze the '1' axis from batching and only store the mask predictions
        pred_masks = [pred['masks'].squeeze(1) for pred in preds]
        pred_masks = [(pred_mask > 0.5).max(0).values.int() for pred_mask in pred_masks] # aggregate all predicted masks into one binary mask
        true_masks = [t['masks'].squeeze(0) for t in targets]

# Visualize masks
for ind in range(len(pred_masks)):
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(true_masks[ind].cpu().numpy())
    plt.axis('off')
    plt.title('Original Mask')
    plt.subplot(1, 2, 2)
    plt.imshow(pred_masks[ind].cpu().numpy())
    plt.axis('off')
    plt.title('Predicted Mask')
    plt.savefig(f'./dsworkflow/src/Lecture_3/masks_{ind}.png')