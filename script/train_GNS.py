""" # noqa: E501
Author: Yuxuan Gu, Gege Wen
Modified by Chunyang Wang
Date: 2025-08-03
Description: Train the GNS model for the 073 dataset.

Example usage:
1. Train the GNS model for the 073 dataset with super-smoothed data
python3 script/train_GNS.py \
   --noise_scale 0.05 \
   --data_path './data/073_C5_r32_ar5_ds8/073_dataset_super_smoothed_ds8_5_autoregressive.pt' \
   --stats_path './data/supersmoothed_train_stats.pt' \
   --epochs 600 \
   --lr 5e-5 \
   --hidden_dim 128 \
   --batch_size 1

2. Train with unsmoothed data
python3 script/train_GNS.py \
   --noise_scale 0.05 \
   --data_path './data/073_C5_r32_ar5_ds8/073_none-smoothed_ds8_5_autoregressive.pt' \
   --stats_path './data/supersmoothed_train_stats.pt' \
   --epochs 600 \
   --lr 5e-5 \
   --hidden_dim 128 \
   --batch_size 1 \
   --tag 'unsmoothed' \

3. Train with smoothed data, no image encoder
python3 script/train_GNS.py \
   --noise_scale 0.05 \
   --data_path './data/073_C5_r32_ar5_ds8/073_dataset_super_smoothed_ds8_5_autoregressive.pt' \
   --stats_path './data/supersmoothed_train_stats.pt' \
   --epochs 600 \
   --lr 5e-5 \
   --hidden_dim 128 \
   --batch_size 1 \
   --tag 'smoothed_no_img_encoder' \
   --image_encoder none \

4. Train the GNS model for the 073 dataset with super-smoothed data and get acceleration
python3 script/train_GNS.py \
   --noise_scale 0.05 \
   --data_path './data/073_C5_r32_ar5_ds8/073_get_acc_ds8_5_autoregressive.pt' \
   --stats_path './data/073_C5_r32_ar5_ds8/073_get_acc_ds8_5_stats.pt' \
   --epochs 600 \
   --lr 5e-5 \
   --hidden_dim 128 \
   --batch_size 1 \
   --tag 'acc' \

"""
import numpy as np
from tqdm import trange
import copy
import os
import wandb
import torch
from pore_net.GNS import GNS
import argparse
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data
from types import SimpleNamespace


# os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_MODE"] = "online"


parser = argparse.ArgumentParser()
parser.add_argument("--noise_scale", type=float, default=0.05)

# Model configuration
parser.add_argument("--num_layers", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--epochs", type=int, default=1200)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--PE", type=str, default="fourier_feature")
parser.add_argument("--image_encoder", type=str, default="cnn")
parser.add_argument("--image_size", type=int, default=32)
parser.add_argument("--opt", type=str, default="adam")
parser.add_argument("--opt_scheduler", type=str, default="cos")
parser.add_argument("--opt_restart", type=int, default=200)
parser.add_argument("--autoregressive_step", type=int, default=1)
parser.add_argument("--tag", type=str, default="")

# Data paths
parser.add_argument(
    "--data_path",
    type=str,
    default=(
        "./data/073_C5_r32_ar5_ds8/"
        "073_dataset_super_smoothed_ds8_5_autoregressive.pt"
    ),
)
parser.add_argument(
    "--stats_path", type=str, default="./data/supersmoothed_train_stats.pt"
)

# Training parameters
parser.add_argument("--opt_decay_step", type=int, default=100)
parser.add_argument("--opt_decay_rate", type=float, default=0.9)

args = parser.parse_args()

# Print all arguments beautifully
print("=" * 80)
print("GNS Training Configuration")
print("=" * 80)
print(f"{'Parameter':<25} {'Value':<55}")
print("-" * 80)

# Print all arguments iteratively
for arg_name, arg_value in vars(args).items():
    print(f"{arg_name:<25} {arg_value:<55}")

print("=" * 80)
print()


def unnormalize(to_unnormalize, mean_vec, std_vec):
    return to_unnormalize * std_vec + mean_vec


def normalize(
    to_normalize,
    mean_vec,
    std_vec,
    node_feature=False,
    label_feature=False,
):  # noqa
    normalized_tensor = (to_normalize - mean_vec) / std_vec
    return normalized_tensor


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == "adam":
        optimizer = optim.Adam(
            filter_fn, lr=args.lr, weight_decay=weight_decay
        )  # noqa E501
    elif args.opt == "sgd":
        optimizer = optim.SGD(
            filter_fn,
            lr=args.lr,
            momentum=0.95,
            weight_decay=weight_decay,  # noqa E501
        )
    elif args.opt == "rmsprop":
        optimizer = optim.RMSprop(
            filter_fn, lr=args.lr, weight_decay=weight_decay
        )  # noqa E501
    elif args.opt == "adagrad":
        optimizer = optim.Adagrad(
            filter_fn, lr=args.lr, weight_decay=weight_decay
        )  # noqa E501
    if args.opt_scheduler == "none":
        return None, optimizer
    elif args.opt_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.opt_decay_step,
            gamma=args.opt_decay_rate,  # noqa E501
        )
    elif args.opt_scheduler == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.opt_restart
        )
    return scheduler, optimizer


# Function to save the model checkpoint
def save_checkpoint(model, optimizer, epoch, filepath):
    # TODO: save the checkpoint in the last epoch (not necessarily the best model) # noqa E501

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, filepath)


# Function to load the model checkpoint
# def load_checkpoint(model, optimizer):
#     print(f"reload from {resume_path}")
#     checkpoint = torch.load(resume_path)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     epoch = checkpoint["epoch"]
#     return epoch


def update_graph(input_data, velocity_pred, mean_vec_y, std_vec_y):

    velocity_pred_unnormalised = unnormalize(
        velocity_pred, mean_vec_y, std_vec_y
    )  # noqa E501
    p_t1 = (
        velocity_pred_unnormalised.clone().detach() + input_data.x[:, :3]
    )  # p_1 = v1 + p0

    # Update input data to the model in the next iteration
    past_velocity = torch.roll(input_data.x[:, 3:], shifts=3, dims=1)
    past_velocity[:, :3] = velocity_pred_unnormalised.clone()

    # Recompute the radius graph
    edge_attribute, edge_index = compute_edge_attr_with_radius_graph(p_t1)

    return Data(
        x=torch.cat((p_t1, past_velocity), dim=1),
        edge_index=edge_index,
        edge_attr=edge_attribute,
        y=input_data.y,
    )


def compute_edge_attr_with_radius_graph(coords_tensor, radius=32):

    edge_index = radius_graph(
        coords_tensor.type(torch.float32),
        r=radius,
        loop=True,
        max_num_neighbors=64,  # noqa E501
    )
    src, dst = edge_index[0], edge_index[1]

    delta = coords_tensor[dst] - coords_tensor[src]

    norms = torch.linalg.norm(delta, dim=1).unsqueeze(1)
    edge_attr_tensor = torch.cat([delta, norms], dim=1)

    return edge_index, edge_attr_tensor


def validation(loader, test_model, stats, config):
    """
    Calculate validation set errors.
    """
    total_loss = 0
    num_loops = 0
    device = config.device

    (
        mean_vec_x,
        std_vec_x,
        mean_vec_edge,
        std_vec_edge,
        mean_vec_y,
        std_vec_y,
    ) = stats
    mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge = (
        mean_vec_x.to(device),
        std_vec_x.to(device),
        mean_vec_edge.to(device),
        std_vec_edge.to(device),
    )
    mean_vec_y, std_vec_y = mean_vec_y.to(device), std_vec_y.to(device)

    test_model.eval()
    with torch.no_grad():
        for data in loader:
            loss = 0
            data = data.to(device)

            for j in range(config.autoregressive_step):
                pred = test_model(
                    data,
                    mean_vec_x,
                    std_vec_x,
                    mean_vec_edge,
                    std_vec_edge,
                )
                loss += test_model.loss(pred, data.y[j, :, :], mean_vec_y, std_vec_y)  # noqa: E501
                data = update_graph(data, pred, mean_vec_y, std_vec_y)

            loss /= config.autoregressive_step
            num_loops += 1
            total_loss += loss

    return total_loss / num_loops


# dataset = torch.load(
#     "dataset_supersmoothed/073_dataset_super_smoothed_5_autoregressive.pt"
# )
# dataset = torch.load(
#     "./data/073_C5_r32_ar5/073_dataset_super_smoothed_5_autoregressive.pt",
#     weights_only=False,
# )


dataset = torch.load(args.data_path, weights_only=False)  # noqa E501
print("dataset_073 loaded, len:", len(dataset))
# dataset.extend(
#     torch.load("dataset_supersmoothed/072_dataset_super_smoothed_5_autoregressive.pt")
# )


# dataset.extend(
#     torch.load(
#         "./data/072_C5_r32_ar5/072_dataset_super_smoothed_5_autoregressive.pt",
#         weights_only=False,
#     )  # noqa E501
# )
# print("dataset_072 loaded, len:", len(dataset))
# dataset.extend(
#     torch.load(
#         "/gpfs/home/cw1722/particle/pore_net/data/072_C5_r32_ar5_ds8/072_dataset_super_smoothed_ds8_5_autoregressive.pt",
#         weights_only=False)  # noqa E501
# )
n_train = int(len(dataset) * 0.75)
print("n_train:", n_train)

print("dataset_all loaded, len:", len(dataset))

# TIF_DATA = torch.load("TIF_DATA_073.pt") | torch.load("TIF_DATA_072.pt")
# print(TIF_DATA.keys())

# resume_dir = "best_models_supersmoothed/latest_checkpoint.pt"
stats_dir = args.stats_path

checkpoint_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "out",
    f"GNS_noise_{args.noise_scale}_073_ds8_{args.tag}",
)

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)


config = SimpleNamespace(
    num_layers=args.num_layers,
    batch_size=args.batch_size,
    hidden_dim=args.hidden_dim,
    epochs=args.epochs,
    weight_decay=args.weight_decay,
    lr=args.lr,
    device=args.device,
    PE=args.PE,
    image_encoder=args.image_encoder,
    image_size=args.image_size,
    opt=args.opt,
    opt_scheduler=args.opt_scheduler,
    opt_restart=args.opt_restart,
    opt_decay_step=args.opt_decay_step,
    opt_decay_rate=args.opt_decay_rate,
    autoregressive_step=args.autoregressive_step,
    noise_scale=args.noise_scale,
)

# Define the model name for saving
device = config.device

# torch_geometric DataLoaders are used for handling the data of lists of graphs
train_loader = DataLoader(
    dataset[:n_train],
    batch_size=config.batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=8,
)
val_loader = DataLoader(
    dataset[n_train:],
    batch_size=config.batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=8,
)

train_stats = torch.load(stats_dir)

(
    mean_vec_x,
    std_vec_x,
    mean_vec_edge,
    std_vec_edge,
    mean_vec_y,
    std_vec_y,
) = train_stats
mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge = (
    mean_vec_x.to(device),
    std_vec_x.to(device),
    mean_vec_edge.to(device),
    std_vec_edge.to(device),
)
mean_vec_y, std_vec_y = mean_vec_y.to(device), std_vec_y.to(device)

# build model
num_node_features = dataset[0].x.shape[1]
num_edge_features = dataset[0].edge_attr.shape[1]
output_dim = 3

model = GNS(
    num_node_features,
    num_edge_features,
    config.hidden_dim,
    output_dim,
    config,
).to(device)
scheduler, opt = build_optimizer(config, model.parameters())
# model.load_state_dict(torch.load(resume_dir)["model_state_dict"])

# train
train_losses = []
val_losses = []
best_val_loss = np.inf
best_loss = np.inf
best_model_train = None
best_model_val = None

start_epoch = 0
print(f"Training starts from epoch {start_epoch} to {config.epochs}.")

best_model_train = copy.deepcopy(model)
best_model_val = copy.deepcopy(model)


wandb.login()
wandb.init(project="GNS_model", config=config)
exp_name = f"GNS_noise_{args.noise_scale}_073_ds8_{args.tag}"
wandb.run.name = exp_name

for epoch in trange(start_epoch, config.epochs, desc="Training", unit="Epochs"):  # noqa: E501
    model_name = "noise-4"
    total_loss = 0
    model.train()
    num_loops = 0

    for batch in train_loader:
        opt.zero_grad()
        batch = batch.to(device)
        loss = 0

        for j in range(config.autoregressive_step):
            exp_id = batch.exp_id.item()
            pred = model(
                batch,
                mean_vec_x,
                std_vec_x,
                mean_vec_edge,
                std_vec_edge,
            )
            loss += model.loss(pred, batch.y[j, :, :], mean_vec_y, std_vec_y)
            batch = update_graph(batch, pred, mean_vec_y, std_vec_y)

        loss.backward()
        opt.step()
        total_loss += loss.item()
        num_loops += 1

    total_loss /= num_loops
    train_losses.append(total_loss)

    wandb.log(
        {
            "Epochs": epoch,
            "Train loss": total_loss,
            "lr": opt.param_groups[0]["lr"],
        }
    )

    if scheduler is not None:
        scheduler.step()

    # For testing purposes. Find the best model that overfits the training data
    if total_loss < best_loss:
        best_loss = total_loss
        best_model_train = copy.deepcopy(model)

    # Validation
    if (epoch + 1) % 1 == 0:
        # We need to modify validation function to handle different experiment IDs as well  # noqa E501
        val_loss = validation(
            val_loader,
            model,
            [
                mean_vec_x,
                std_vec_x,
                mean_vec_edge,
                std_vec_edge,
                mean_vec_y,
                std_vec_y,
            ],
            config,
        ).item()
        val_losses.append(val_loss)

        # saving model
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        # save the model if the current one is better than the previous best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_val = copy.deepcopy(model)

        PATH_train = os.path.join(checkpoint_dir, model_name + "_train.pt")
        PATH_val = os.path.join(checkpoint_dir, model_name + "_val.pt")

        # Add safety checks before saving models
        if best_model_train is not None:
            torch.save(best_model_train.state_dict(), PATH_train)
        else:
            print(
                "Warning: best_model_train is None, saving current model instead"  # noqa E501
            )
            torch.save(model.state_dict(), PATH_train)

        if best_model_val is not None:
            torch.save(best_model_val.state_dict(), PATH_val)
        else:
            print("Warning: best_model_val is None, saving current model instead")  # noqa E501
            torch.save(model.state_dict(), PATH_val)

        wandb.log({"Epochs": epoch, "Val loss": val_loss})

        # save the latest model to resume training if needed
        checkpt_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        save_checkpoint(model, opt, epoch, checkpt_path)

        print(
            "train loss: ",
            str(round(total_loss, 5)),
            ", validation loss: ",
            str(round(val_loss, 5)),
            "lr: ",
            str(round(opt.param_groups[0]["lr"], 5)),
        )
    else:
        val_losses.append("-")
