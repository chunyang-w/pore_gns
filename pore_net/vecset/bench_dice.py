"""
Benchmark for vecset model on dice score.

Written by: Chunyang Wang
GitHub username: chunyang-w
Example usage:

1. On HX1 server, run:
python pore_net/vecset/bench_dice.py \
    --data_dir ./data/PC_AE_073_segmented_tifs_pad_size20_down10_dist_thresh5.0/ \
    --checkpoints ./out/ae/ae_d512_m512/checkpoint-1320.pth \
    --model_name ae_d512_m512 \
    --data_start 150 \
    --data_end 200 \

1.2 On HX1 server, run: (This is on down 8 dataset)
python pore_net/vecset/bench_dice.py \
    --data_dir ./data/PC_AE_073_segmented_tifs_pad_size20_down8_dist_thresh5.0/ \
    --checkpoints ./out/ae/ae_d512_m512/checkpoint-3350.pth \
    --model_name ae_d512_m512 \
    --data_start 150 \
    --data_end 200 \

2. On Chunyang's laptop, run:
python pore_net/vecset/bench_dice.py \
    --data_dir /Users/chunyang/projects/particle/ditto/data/073_segmented_tifs_pad_size20_down10dist_thresh5/ \
    --checkpoints /Users/chunyang/Downloads/checkpoint-1320.pth \
    --model_name ae_d512_m512 \
    --data_start 150 \
    --data_end 200 \
"""

# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import os
import torch
import argparse
import pore_net.vecset.model as models_ae
import pandas as pd

from pore_net.vecset.dataset import PoreShapeDataset
from pore_net.utils import get_dice_score
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIS = False

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--checkpoints", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--data_start", type=int, default=150)
parser.add_argument("--data_end", type=int, default=200)


if __name__ == "__main__":
    # Add argparse.Namespace to safe globals
    # Add box_size argument
    args = parser.parse_args()
    print("\n" + "=" * 50)
    print("VecSet Model Benchmark")
    print("=" * 50)
    print(f"Model name:     {args.model_name}")
    print(f"Data range:     {args.data_start} - {args.data_end}")
    print(f"Checkpoint:     {args.checkpoints}")
    print(f"Data dir:       {args.data_dir}")
    print(f"Device:         {device}")
    print("=" * 50 + "\n")

    checkpoint_base_dir = os.path.dirname(args.checkpoints)
    checkpoint_base_dir = os.path.abspath(checkpoint_base_dir)
    csv_path = os.path.join(checkpoint_base_dir, "dice_scores.csv")
    print("checkpoint_base_dir", checkpoint_base_dir)

    # Load the dataset
    dataset_pd = PoreShapeDataset(  # Sample points from all points
        data_dir=args.data_dir, transform=None, split="val"
    )
    # Plot ground truth
    dataset_gt = PoreShapeDataset(  # All points are retained
        data_dir=args.data_dir,
        sampling=False,
        split="val",
        transform=None,
    )

    print("dataset_pd", len(dataset_pd))
    print("dataset_gt", len(dataset_gt))

    model = models_ae.__dict__[args.model_name]()
    model.to(device)
    model.eval()
    model.load_state_dict(
        torch.load(args.checkpoints, map_location=device, weights_only=False)["model"],
        strict=True,
    )  # noqa: E501

    # Initialize list to store results
    results = []

    for data_idx in tqdm(range(args.data_start, args.data_end)):
        sample_gt = dataset_gt[data_idx]
        points_gt, labels_gt, surface_gt = sample_gt
        labels_gt = labels_gt.numpy()

        sample_pred = dataset_pd[data_idx]
        points_pred, _, surface_pred = sample_pred

        points_pred = points_pred.unsqueeze(0).to(device)
        surface_pred = surface_pred.unsqueeze(0).to(device)

        print("points_gt.shape", points_gt.shape)

        # If inference size is too large, split into chunks
        n_chunks = 4
        pred_chunks = []
        points_gt_chunks = torch.chunk(points_gt, n_chunks, dim=0)

        with torch.no_grad():
            # points_gt is the ground truth points
            # Because it is not sampled, it is essentially all
            # the points in the box, represented as coords
            for points_gt_chunk in points_gt_chunks:
                points_gt_chunk = points_gt_chunk.unsqueeze(0).to(device)
                pred = model(surface_pred, points_gt_chunk)["logits"]
                pred = pred.cpu()
                print("pred.shape", pred.shape)
                pred_chunks.append(pred)
            pred = torch.cat(pred_chunks, dim=1)

        cube_pred = (pred > -0).numpy().reshape(159, 201, 201).transpose(1, 2, 0)
        cube_gt = labels_gt.reshape(159, 201, 201).transpose(1, 2, 0)

        dice_score = get_dice_score(cube_pred, cube_gt)
        print(f"Dice score: {dice_score}")

        # Append results
        results.append({"data_idx": data_idx, "dice_score": dice_score})

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
