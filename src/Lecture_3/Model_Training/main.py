"""
We will train a Mask R-CNN to segment bees in images.
Original paper: https://arxiv.org/abs/1703.06870
Check implementation details in PyTorch here: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html

Please download the dataset from Google Drive:
https://drive.google.com/drive/folders/1GocUmRN6Pb2XhojTHJPFRBV7OIc4nNW_

Check Lecture_2/DSW_Data_Acquisition.ipynb for details on how to download the dataset.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm # for progress bar
from dataset import BeeDataset, collate_fn
from modelling import get_maskrcnn
from metrics import compute_mean_iou


def main():
    ### Load the data
    # Define paths
    train_img_dir   = "dsworkflow/data/Bees_train"
    train_masks_dir = "dsworkflow/data/Bees_masks_train"
    val_img_dir     = "dsworkflow/data/Bees_val"
    val_masks_dir   = "dsworkflow/data/Bees_masks_val"
    
    # Create Datasets
    train_dataset = BeeDataset(train_img_dir, train_masks_dir)
    val_dataset   = BeeDataset(val_img_dir, val_masks_dir)

    # Create DataLoaders
    batch_size = 4 # Adjust according to your GPU memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)


    ### Define the model
    model = get_maskrcnn()

    ### Training
    lr, weight_decay = 1e-3, 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.1, min_lr=lr*0.01)
    num_epochs = 10
    
    for epoch in range(num_epochs):
        print(f" Epoch {epoch+1} starting")

        # Training
        model.train()
        train_loss = 0
        train_loader_tqdm = tqdm(train_loader, desc="Training", leave=True, position=0)
        for imgs, targets in train_loader_tqdm:
            imgs = [img.to("cuda") for img in imgs]
            targets = [{k: v.to("cuda") for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = torch.sum(torch.stack(list(loss_dict.values())))  # total loss (check with the debugger what individual losses are computed)
            
            optimizer.zero_grad()
            losses.backward() # backprop
            optimizer.step()  # SGD update
            """
            Important note on PyTorch vs Tensorflow:
            In TF, the optimizer and the loss that it tries to minize communicate via the model.compile() method. The backward pass and the forward update happen both within the TF graph that is defined after the model compilation.
            In PT, notice that the loss and the optimizer do not communicate directly, like optimizer.step(loss) or similarly. Instead, PT uses a background computational graph that keeps track of all the gradients.
            As long as all inputs and model parameters are PT tensors with attributes .grad and .requires_grad=True, the loss and optimizer will be able to access that information.
            """

            train_loss += losses.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1} training loss: {train_loss}")

        # Validation
        print("Running evaluation...")
        train_iou = compute_mean_iou(model, train_loader)
        val_iou   = compute_mean_iou(model, val_loader)
        print(f"Epoch {epoch+1} - Mean IoU (Train): {train_iou}, Mean IoU (Validation): {val_iou}")

        # Update LR
        lr_scheduler.step(val_iou)

    print("Training finished.")
    # Save the trained weights
    torch.save(model.state_dict(), "dsworkflow/models/maskrcnn.pt")


if __name__ == "__main__":
    main()