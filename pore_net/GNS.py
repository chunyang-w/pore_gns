# Code from Yuxuan Gu https://github.com/guyuxuan9
# modified by Chunyang Wang
import torch
import torch_scatter
import torch.nn as nn

from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch_geometric.nn.conv import MessagePassing
from pore_net.CNN import CNNEncoder


def unnormalize(to_unnormalize, mean_vec, std_vec):
    return to_unnormalize * std_vec + mean_vec


def normalize(
    to_normalize,
    mean_vec,
    std_vec,
    node_feature=False,
    label_feature=False,
):  # noqa: E501
    normalized_tensor = (to_normalize - mean_vec) / std_vec
    return normalized_tensor


class GNS(torch.nn.Module):
    def __init__(
        self,
        input_dim_node,
        input_dim_edge,
        hidden_dim,
        output_dim,
        args,
        emb=False,  # noqa E501
    ):
        super(GNS, self).__init__()
        self.args = args
        self.num_layers = args.num_layers
        self.PE = args.PE
        self.image_encoder = args.image_encoder

        # encoder convert raw inputs into latent embeddings
        self.node_encoder = Sequential(
            Linear(input_dim_node, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim),
        )

        if args.image_encoder == "cnn":
            self.cnn_encoder = CNNEncoder(
                num_filters=hidden_dim, image_size=args.image_size
            )

        self.edge_encoder = Sequential(
            Linear(input_dim_edge, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim),
        )

        self.processor = nn.ModuleList()
        assert self.num_layers >= 1, "Number of message passing layers is not >=1"

        processor_layer = self.build_processor_model()
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(hidden_dim, hidden_dim))

        # decoder: only for node embeddings
        self.decoder = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, output_dim),
        )

    def build_processor_model(self):
        return ProcessorLayer

    def forward(self, data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge):
        x, edge_index, edge_attr, image_3D = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.image_3D,
        )

        # Debug: Print shapes before normalization
        # print("Debug - Forward method shapes before normalization:")
        # print(f"x shape: {x.shape}")
        # print(f"mean_vec_x shape: {mean_vec_x.shape}")
        # print(f"std_vec_x shape: {std_vec_x.shape}")
        # print(f"edge_attr shape: {edge_attr.shape}")
        # print(f"mean_vec_edge shape: {mean_vec_edge.shape}")
        # print(f"std_vec_edge shape: {std_vec_edge.shape}")

        x = normalize(x, mean_vec_x, std_vec_x, node_feature=True)
        if self.args.noise_scale is not None:
            noise = torch.randn_like(x) * self.args.noise_scale
            x = x + noise
        else:
            x = x

        edge_attr = normalize(edge_attr, mean_vec_edge, std_vec_edge)

        # Step 1: encode node/edge features into latent node/edge embeddings
        x_features = self.node_encoder(x)  # (num_nodes, hidden_dim)
        if self.image_encoder == "cnn":
            image_features = self.cnn_encoder(image_3D)
        elif self.image_encoder == "vit":
            image_features = self.vit_encoder(image_3D)
        elif self.image_encoder == "none":
            # print("No image encoder")
            pass
        if self.image_encoder != "none":
            x = x_features + image_features  # (num_nodes, hidden_dim)
        else:
            x = x_features

        edge_attr = self.edge_encoder(
            edge_attr
        )  # output shape is the specified hidden dimension

        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_layers):
            x, edge_attr = self.processor[i](x, edge_index, edge_attr)

        # step 3: decode latent node embeddings into physical quantities of interest  # noqa E501
        return self.decoder(x)

    def loss(self, pred, gt, mean_vec_y, std_vec_y):
        labels = normalize(gt, mean_vec_y, std_vec_y, label_feature=True)

        # Find squared errors for the masked values
        squared_error_oil = (labels - pred) ** 2

        mean_squared_error = torch.mean(squared_error_oil)
        loss = torch.sqrt(mean_squared_error)

        return loss


class ProcessorLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ProcessorLayer, self).__init__(**kwargs)
        self.edge_mlp = Sequential(
            Linear(3 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels),
        )

        self.node_mlp = Sequential(
            Linear(2 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr, size=None):
        out, updated_edges = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=size
        )  # out has the shape of [E, out_channels]
        updated_nodes = torch.cat(
            [x, out], dim=1
        )  # Complete the aggregation through self-aggregation
        updated_nodes = x + self.node_mlp(updated_nodes)  # residual connection

        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        updated_edges = torch.cat(
            [x_i, x_j, edge_attr], dim=1
        )  # tmp_emb has the shape of [E, 3 * in_channels]
        updated_edges = self.edge_mlp(updated_edges) + edge_attr
        return updated_edges

    def aggregate(self, updated_edges, edge_index, dim_size=None):
        out = torch_scatter.scatter(
            updated_edges, edge_index[1, :], dim=0, reduce="sum"
        )
        return out, updated_edges


# class GNS(torch.nn.Module):
#     def __init__(
#         self,
#         input_dim_node,
#         input_dim_edge,
#         hidden_dim,
#         output_dim,
#         args,
#         emb=False,
#     ):
#         super(GNS, self).__init__()
#         self.num_layers = args.num_layers
#         self.PE = args.PE
#         self.image_encoder = args.image_encoder

#         # encoder convert raw inputs into latent embeddings
#         self.node_encoder = Sequential(
#             Linear(input_dim_node, hidden_dim),
#             ReLU(),
#             Linear(hidden_dim, hidden_dim),
#             LayerNorm(hidden_dim),
#         )

#         if args.image_encoder == "cnn":
#             self.cnn_encoder = CNNEncoder(
#                 num_filters=hidden_dim, image_size=args.image_size
#             )

#         self.edge_encoder = Sequential(
#             Linear(input_dim_edge, hidden_dim),
#             ReLU(),
#             Linear(hidden_dim, hidden_dim),
#             LayerNorm(hidden_dim),
#         )

#         self.processor = nn.ModuleList()
#         assert (
#             self.num_layers >= 1
#         ), "Number of message passing layers is not >=1"

#         processor_layer = self.build_processor_model()
#         for _ in range(self.num_layers):
#             self.processor.append(processor_layer(hidden_dim, hidden_dim))

#         # decoder: only for node embeddings
#         self.decoder = Sequential(
#             Linear(hidden_dim, hidden_dim),
#             ReLU(),
#             Linear(hidden_dim, output_dim),
#         )

#     def build_processor_model(self):
#         return ProcessorLayer

#     def forward(
#         self, data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge
#     ):
#         x, edge_index, edge_attr, image_3D = (
#             data.x,
#             data.edge_index,
#             data.edge_attr,
#             data.image_3D,
#         )

#         # Debug: Print shapes before normalization
#         # print("Debug - Forward method shapes before normalization:")
#         # print(f"x shape: {x.shape}")
#         # print(f"mean_vec_x shape: {mean_vec_x.shape}")
#         # print(f"std_vec_x shape: {std_vec_x.shape}")
#         # print(f"edge_attr shape: {edge_attr.shape}")
#         # print(f"mean_vec_edge shape: {mean_vec_edge.shape}")
#         # print(f"std_vec_edge shape: {std_vec_edge.shape}")

#         x = normalize(x, mean_vec_x, std_vec_x, node_feature=True)
#         edge_attr = normalize(edge_attr, mean_vec_edge, std_vec_edge)

#         # Step 1: encode node/edge features into latent node/edge embeddings
#         x_features = self.node_encoder(x)  # (num_nodes, hidden_dim)
#         if self.image_encoder == "cnn":
#             image_features = self.cnn_encoder(image_3D)
#         elif self.image_encoder == "vit":
#             image_features = self.vit_encoder(image_3D)
#         x = x_features + image_features  # (num_nodes, hidden_dim)

#         edge_attr = self.edge_encoder(
#             edge_attr
#         )  # output shape is the specified hidden dimension

#         # step 2: perform message passing with latent node/edge embeddings
#         for i in range(self.num_layers):
#             x, edge_attr = self.processor[i](x, edge_index, edge_attr)

#         # step 3: decode latent node embeddings into physical quantities of interest  # noqa E501
#         return self.decoder(x)

#     def loss(self, pred, gt, mean_vec_y, std_vec_y):
#         labels = normalize(gt, mean_vec_y, std_vec_y, label_feature=True)

#         # Find squared errors for the masked values
#         squared_error_oil = (labels - pred) ** 2

#         mean_squared_error = torch.mean(squared_error_oil)
#         loss = torch.sqrt(mean_squared_error)

#         return loss


# class ProcessorLayer(MessagePassing):
#     def __init__(self, in_channels, out_channels, **kwargs):
#         super(ProcessorLayer, self).__init__(**kwargs)
#         self.edge_mlp = Sequential(
#             Linear(3 * in_channels, out_channels),
#             ReLU(),
#             Linear(out_channels, out_channels),
#             LayerNorm(out_channels),
#         )

#         self.node_mlp = Sequential(
#             Linear(2 * in_channels, out_channels),
#             ReLU(),
#             Linear(out_channels, out_channels),
#             LayerNorm(out_channels),
#         )

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.edge_mlp[0].reset_parameters()
#         self.edge_mlp[2].reset_parameters()

#         self.node_mlp[0].reset_parameters()
#         self.node_mlp[2].reset_parameters()

#     def forward(self, x, edge_index, edge_attr, size=None):
#         out, updated_edges = self.propagate(
#             edge_index, x=x, edge_attr=edge_attr, size=size
#         )  # out has the shape of [E, out_channels]
#         updated_nodes = torch.cat(
#             [x, out], dim=1
#         )  # Complete the aggregation through self-aggregation
#         updated_nodes = x + self.node_mlp(updated_nodes)  # residual connection
#         return updated_nodes, updated_edges

#     def message(self, x_i, x_j, edge_attr):
#         updated_edges = torch.cat(
#             [x_i, x_j, edge_attr], dim=1
#         )  # tmp_emb has the shape of [E, 3 * in_channels]
#         updated_edges = self.edge_mlp(updated_edges) + edge_attr
#         return updated_edges

#     def aggregate(self, updated_edges, edge_index, dim_size=None):
#         out = torch_scatter.scatter(
#             updated_edges, edge_index[1, :], dim=0, reduce="sum"
#         )
#         return out, updated_edges
