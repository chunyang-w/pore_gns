"""
Written by: Chunyang Wang
GitHub username: chunyang-w

Code to train a FNO model for pore reconstruction.

Example usage:
python pore_net/fno/train.py \
    --data_dir ./data/tif_073_segmented_tifs_down8/ \
    --batch_size 4 \
    --output_dir ./out/fno/ \
    --epochs 100 \
    --lr 1e-3 \
    --modes 8 \

"""

import os
import torch
import argparse
import wandb

from torch.utils.data import DataLoader
from pore_net.fno.model import FNO3d
from pore_net.unet_recon.dataset import PoreShapeDatasetUnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_wandb = True


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    avg_loss = 0.0
    avg_dice = 0.0
    for batch in loader:
        in_data = batch
        in_data = in_data.to(device)
        optimizer.zero_grad()
        pred = model(in_data)
        loss = criterion(pred, in_data)
        avg_loss += loss.item()

        # Calculate dice score
        dice_score = calculate_dice_score(pred, in_data)
        avg_dice += dice_score

        loss.backward()
        optimizer.step()

    avg_loss /= len(loader)
    avg_dice /= len(loader)
    return avg_loss, avg_dice, model


def calculate_dice_score(pred, target, threshold=0.5):
    """
    Calculate dice score between predictions and targets.
    Args:
        pred: predicted tensor
        target: target tensor
        threshold: threshold for binarization
    Returns:
        dice score
    """
    with torch.no_grad():
        # Apply sigmoid to predictions and binarize
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()

        # Calculate dice coefficient
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum()

        # Avoid division by zero
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        dice = (2.0 * intersection) / union
        return dice.item()


def test_one_epoch(model, loader, criterion):
    model.eval()
    avg_loss = 0.0
    avg_dice = 0.0
    with torch.no_grad():
        for batch in loader:
            in_data = batch
            in_data = in_data.to(device)
            pred = model(in_data)
            loss = criterion(pred, in_data)
            avg_loss += loss.item()
            # Calculate dice score
            dice_score = calculate_dice_score(pred, in_data)
            avg_dice += dice_score

    avg_loss /= len(loader)
    avg_dice /= len(loader)
    return avg_loss, avg_dice


def train(epoch, model, optimizer, loader_train, loader_test, output_dir):
    if use_wandb:
        wandb.init(project="pore_fno", name="fno_recon")
    criterion = torch.nn.MSELoss()
    model = model.to(device)
    for i in range(epoch):
        loss, dice, model = train_one_epoch(
            model=model, loader=loader_train, optimizer=optimizer, criterion=criterion
        )  # noqa: E501
        print(f"Epoch \t{i+1}, Train Loss \t{loss}, Dice Score \t{dice}")
        if use_wandb:
            wandb.log({"train_loss": loss, "dice_score": dice})
        if (i + 1) % 50 == 0:
            info, test_dice = test_one_epoch(
                model=model, loader=loader_test, criterion=criterion
            )
            print(
                f"\t Epoch \t{i+1}, Test Loss \t{info}, "
                f"Test Dice Score \t{test_dice}"
            )
            if use_wandb:
                wandb.log({"test_loss": info, "test_dice_score": test_dice})
            model_path = os.path.join(output_dir, f"epoch_{i+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(model_path, "saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/PC_073_segmented_tifs_pad_0_ds8near_thresh3.0/",
    )
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--output_dir", type=str, default="./out/fno")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--modes", type=int, default=8)
    args = parser.parse_args()

    # Print arguments in a formatted way
    print("\n" + "=" * 50)
    print("Training FNO with the following settings:")
    print("-" * 50)
    for arg, value in vars(args).items():
        print(f"{arg:15s}: {value}")
    print("=" * 50 + "\n")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    model = FNO3d(modes1=args.modes, modes2=args.modes, modes3=args.modes, width=36)

    train_dataset = PoreShapeDatasetUnet(
        data_dir=args.data_dir,
        frame_idxs=[i for i in range(150)],
    )
    test_set = PoreShapeDatasetUnet(
        data_dir=args.data_dir,
        frame_idxs=[i for i in range(150, 200)],
    )

    loader_train = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
    loader_test = DataLoader(test_set, batch_size=args.batch_size, num_workers=4)

    # Init optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-6,
    )

    train(
        epoch=args.epochs,
        model=model,
        optimizer=optimizer,
        loader_train=loader_train,
        loader_test=loader_test,
        output_dir=args.output_dir,
    )
