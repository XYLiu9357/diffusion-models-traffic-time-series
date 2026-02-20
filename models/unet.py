import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SpatioTemporalUNet(nn.Module):
    """
    Shared U-Net backbone with:
    - Temporal convolutions (along time dimension)
    - Graph convolutions (along sensor dimension)
    - Skip connections for U-Net structure
    - Proper handling of multi-resolution skips
    """

    def __init__(
        self,
        adj_mx: np.ndarray,
        in_channels: int = 1,
        time_steps: int = 12,
        num_sensors: int = 207,
        hidden_dims: List[int] = [64, 128, 256],
        use_gat: bool = False,
        debug: bool = True,
    ):
        super().__init__()
        self.adj_mx = adj_mx
        self.time_steps = time_steps
        self.num_sensors = num_sensors
        self.hidden_dims = hidden_dims
        self.debug = debug

        self.register_buffer("adj_mx_tensor", torch.FloatTensor(adj_mx))

        # Initial projection: [B, F, T, N] -> [B, H1, T, N]
        self.input_proj = nn.Conv2d(
            in_channels, hidden_dims[0], kernel_size=(3, 3), padding=(1, 1)
        )

        # Encoder (downsampling)
        self.encoders = nn.ModuleList()
        self.graph_convs_enc = nn.ModuleList()

        in_dim = hidden_dims[0]
        current_time = time_steps
        for h_dim in hidden_dims[1:]:
            # Temporal downsampling (halves time)
            self.encoders.append(
                nn.Conv2d(
                    in_dim, h_dim, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)
                )
            )
            self.graph_convs_enc.append(GraphConvBlock(h_dim, h_dim, adj_mx, use_gat))
            in_dim = h_dim
            current_time = current_time // 2

        # Bottleneck (no further downsampling)
        self.bottleneck_time = nn.Conv2d(
            hidden_dims[-1], hidden_dims[-1], kernel_size=(3, 1), padding=(1, 0)
        )
        self.bottleneck_graph = GraphConvBlock(
            hidden_dims[-1], hidden_dims[-1], adj_mx, use_gat
        )

        # Decoder (upsampling) – we will have len(hidden_dims)-1 decoder blocks
        self.decoders = nn.ModuleList()
        self.graph_convs_dec = nn.ModuleList()
        self.skip_convs = (
            nn.ModuleList()
        )  # project skip channels to match decoder channels
        self.decoder_proj = nn.ModuleList()  # project concatenated features back

        # Build decoder in reverse order (from lowest res to highest)
        # reversed_dims = [256, 128, 64] (if hidden_dims = [64,128,256])
        reversed_dims = hidden_dims[::-1]
        in_dim = reversed_dims[0]  # bottleneck channels

        for i, out_dim in enumerate(reversed_dims[1:]):
            # Temporal upsampling (doubles time)
            self.decoders.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size=(3, 1),
                    stride=(2, 1),
                    padding=(1, 0),
                    output_padding=(1, 0),
                )
            )

            # The corresponding encoder skip should have the same resolution as the decoder after upsampling.
            # The encoder skip at this resolution comes from the layer with index (len(hidden_dims)-2-i).
            # Its channel count is hidden_dims[len(hidden_dims)-2-i] (or hidden_dims[i] for the first decoder?).
            # Simpler: we will dynamically match dimensions in forward; here we just create a skip_conv that
            # can project any input channels to out_dim. We'll use a 1x1 conv with flexible in_channels later.
            # To avoid hardcoding, we create a placeholder that will be replaced in forward? Not possible.
            # So we must know the exact input channels for each skip.
            # The skip at this stage comes from encoder output after i-th encoder block (counting from top).
            # For i=0 (first decoder), we need skip from after first encoder (which has channels = hidden_dims[1] = 128).
            # For i=1 (second decoder), we need skip from input_proj (channels = hidden_dims[0] = 64).
            # Thus we can compute:
            skip_channels = hidden_dims[
                len(hidden_dims) - 2 - i
            ]  # when i=0: hidden_dims[1]=128; i=1: hidden_dims[0]=64
            self.skip_convs.append(nn.Conv2d(skip_channels, out_dim, kernel_size=1))

            # After concatenation, we have 2*out_dim channels
            self.decoder_proj.append(nn.Conv2d(out_dim * 2, out_dim, kernel_size=1))

            self.graph_convs_dec.append(
                GraphConvBlock(out_dim, out_dim, adj_mx, use_gat)
            )

            in_dim = out_dim

        # Output projection: back to original feature dimension
        self.output_proj = nn.Conv2d(
            hidden_dims[0], in_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )

        if self.debug:
            self._print_architecture()

    def _print_architecture(self):
        print("\n" + "=" * 60)
        print("SPATIO-TEMPORAL U-NET ARCHITECTURE (FIXED)")
        print("=" * 60)
        print(f"Input: [batch_size, {self.time_steps}, {self.num_sensors}, 1]")
        print(f"Hidden dimensions: {self.hidden_dims}")
        print(
            f"Using GAT: {hasattr(self.graph_convs_enc[0], 'use_gat') and self.graph_convs_enc[0].use_gat}"
        )
        print("\n--- Skip Connections ---")
        print("  Skip 0: after input_proj (time=12, channels=64)")
        print("  Skip 1: after encoder block 0 (time=6, channels=128)")
        print("  Skip 2: after encoder block 1 (time=3, channels=256)")
        print("\n--- Decoder ---")
        print("  Decoder 0: 3 -> 6, uses Skip 1 (128 -> 128)")
        print("  Decoder 1: 6 -> 12, uses Skip 0 (64 -> 64)")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\nTotal parameters: {total_params:,}")
        print("=" * 60)

    def forward(self, x, timesteps, cond=None):
        batch_size = x.shape[0]

        # Reshape to [B, F, T, N]
        x = x.permute(0, 3, 1, 2).contiguous()

        # Time embedding
        t_emb = self.time_embed(timesteps.float().unsqueeze(-1))  # [B, H]
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

        # Initial projection
        h = self.input_proj(x)
        h = h + t_emb

        # Store skip connections: start with input_proj output (highest resolution)
        skips = [h]  # list of tensors [B, C, T, N] with decreasing resolution

        # === ENCODER ===
        for encoder, graph_conv in zip(self.encoders, self.graph_convs_enc):
            h = encoder(h)  # temporal down
            # Prepare for graph conv
            h = h.permute(0, 2, 3, 1).contiguous()  # [B, T, N, C]
            h = graph_conv(h)  # spatial processing
            h = h.permute(0, 3, 1, 2).contiguous()  # back to [B, C, T, N]
            skips.append(h)  # store after each encoder block

        # === BOTTLENECK ===
        h = self.bottleneck_time(h)
        h = h.permute(0, 2, 3, 1).contiguous()
        h = self.bottleneck_graph(h)
        h = h.permute(0, 3, 1, 2).contiguous()

        # === DECODER ===
        # We now have skips: [skip0 (original res), skip1 (after enc0), skip2 (after enc1)]
        # Decoder starts from bottleneck (skip2 resolution). We will pop skips from the end (excluding the last one)
        # to match the increasing resolution.
        # For each decoder block, we need the skip that has the same temporal size as the decoder after upsampling.
        # We'll iterate through decoder blocks and pick the appropriate skip from the list based on time dimension.
        for i, (decoder, graph_conv, skip_conv, proj_conv) in enumerate(
            zip(self.decoders, self.graph_convs_dec, self.skip_convs, self.decoder_proj)
        ):
            # Temporal upsampling
            h = decoder(h)  # [B, C_out, T_new, N]

            # Find the skip connection with matching temporal size
            # We search from the end of skips list (excluding the last one which is the bottleneck itself)
            matching_skip = None
            for skip in reversed(
                skips[:-1]
            ):  # skip the bottleneck (last) because we already have it
                if skip.shape[2] == h.shape[2]:
                    matching_skip = skip
                    break

            if matching_skip is None:
                raise RuntimeError(
                    f"No skip connection found with time dimension {h.shape[2]}"
                )

            # Project skip to match decoder channels
            skip_proj = skip_conv(matching_skip)

            # Concatenate along channel dimension
            h = torch.cat([h, skip_proj], dim=1)
            h = proj_conv(h)

            # Spatial graph conv
            h = h.permute(0, 2, 3, 1).contiguous()
            h = graph_conv(h)
            h = h.permute(0, 3, 1, 2).contiguous()

        # === OUTPUT ===
        out = self.output_proj(h)
        out = out.permute(0, 2, 3, 1).contiguous()  # [B, T, N, F]

        return out


class GraphConvBlock(nn.Module):
    """Graph convolution block for spatial processing with proper batching."""

    def __init__(self, in_channels, out_channels, adj_mx, use_gat=False, debug=False):
        super().__init__()
        self.adj_mx = adj_mx
        self.use_gat = use_gat
        self.debug = debug
        self.num_nodes = adj_mx.shape[0]

        # Base edge_index for a single graph (without offsets)
        self.register_buffer("base_edge_index", self._adj_to_edge_index(adj_mx))

        if use_gat:
            self.conv = GATConv(in_channels, out_channels, heads=4, concat=False)
        else:
            self.conv = GCNConv(in_channels, out_channels)

        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.ReLU()

    def _adj_to_edge_index(self, adj_mx):
        """Convert dense adjacency to edge_index for PyG."""
        adj = torch.FloatTensor(adj_mx)
        # Get indices of non-zero entries (including self-loops if any)
        edge_index = adj.nonzero().t().contiguous()
        if self.debug:
            print(f"  Created base edge_index with {edge_index.shape[1]} edges")
        return edge_index

    def forward(self, x):
        """
        Args:
            x: [batch, time, nodes, channels]
        Returns:
            out: [batch, time, nodes, out_channels]
        """
        if self.debug:
            print(f"    GraphConvBlock input: {x.shape}")

        batch, time, nodes, channels = x.shape
        device = x.device
        num_graphs = batch * time
        nodes_per_graph = self.num_nodes

        # Flatten node features: [batch*time*nodes, channels]
        x_flat = x.view(-1, channels)  # [total_nodes, channels]

        # Build batched edge_index with offsets
        E = self.base_edge_index.shape[1]
        # Repeat base edge_index for each graph
        batched_edge_index = self.base_edge_index.repeat(
            1, num_graphs
        )  # [2, E * num_graphs]

        # Create offsets for each graph
        offsets = (
            torch.arange(num_graphs, device=device) * nodes_per_graph
        )  # [num_graphs]
        # For each edge, determine which graph it belongs to
        graph_id = torch.arange(num_graphs, device=device).repeat_interleave(
            E
        )  # [E * num_graphs]
        offset_per_edge = offsets[graph_id]  # [E * num_graphs]

        # Add offsets to both rows of edge_index
        batched_edge_index = batched_edge_index + offset_per_edge.unsqueeze(
            0
        ).expand_as(batched_edge_index)

        # Apply graph convolution
        out = self.conv(x_flat, batched_edge_index)  # [total_nodes, out_channels]

        # Reshape back to [batch, time, nodes, out_channels]
        out = out.view(batch, time, nodes, -1)

        out = self.norm(out)
        out = self.activation(out)

        if self.debug:
            print(f"    GraphConvBlock output: {out.shape}")

        return out
