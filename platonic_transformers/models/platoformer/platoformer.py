import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from platonic_transformers.models.platoformer.block import PlatonicBlock
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from platonic_transformers.models.platoformer.linear import PlatonicLinear
from platonic_transformers.models.platoformer.io import pool, to_scalars_vectors
from platonic_transformers.models.platoformer.patchifiers import (
    StandardPatchifier, EdgeConvPatchifier,
)


class PlatonicTransformer(nn.Module):
    """
    A Transformer architecture equivariant to the symmetries of a specified Platonic solid.

    This model processes point cloud data. It first embeds input node features, then
    "lifts" them into a group-equivariant feature space. A series of PlatonicBlocks
    process these features equivariantly. Finally, for graph-level tasks, it pools
    over the nodes and the group to produce a single invariant prediction. For node-level
    tasks, it pools over the group axis to produce invariant node predictions.

    Args:
        input_dim (int): Dimensionality of the initial node features.
        hidden_dim (int): The per-group-element channel dimension used throughout the model.
        output_dim (int): Dimensionality of the final output.
        nhead (int): Number of attention heads in each PlatonicBlock.
        num_layers (int): Number of PlatonicBlock layers.
        solid_name (str): The name of the Platonic solid ('tetrahedron', 'octahedron',
                          'icosahedron') to define the symmetry group.
        ffn_dim_factor (int): Multiplier for the feed-forward network's hidden dimension,
                              relative to `hidden_dim`.
        scalar_task_level (str): "node" or "graph". Determines the pooling strategy.
        dropout (float): Dropout rate.
        drop_path_rate (float): Stochastic depth rate. Default: 0.0.
        layer_scale_init_value (Optional[float]): Initial value for LayerScale. Default: None.
        **kwargs: Additional keyword arguments for the PlatonicBlock layers
    """
    def __init__(self,
        # Basic/essential specification:
        input_dim: int,
        input_dim_vec: int,
        hidden_dim: int,
        output_dim: int,
        output_dim_vec: int,
        nhead: int,
        num_layers: int,
        solid_name: str,
        spatial_dim: int = 3,
        dense_mode: bool = False, # force dense mode, even if batch is provided
        # Pooling and readout specification:
        scalar_task_level: str = "graph",
        vector_task_level: str = "node",
        ffn_readout: bool = True,
        trivial_readout: bool = False,  # symmetry-break: pool group, then standard linears
        # Attention block specification:
        mean_aggregation: bool = False,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: Optional[float] = None,
        attention: bool = False,
        ffn_dim_factor: int = 4,
        norm_type: str = "layernorm",
        # RoPE and APE specification:
        rope_sigma: float = 1.0,  # if None it is not used
        ape_sigma: float = None,  # if None it is not used
        learned_freqs: bool = True,
        freq_init: str = 'random',
        use_key: bool = False,
        rope_on_values: bool = False,
        rope_v_separate_freqs: bool = False,
        attention_backend: str = "scatter",  # "scatter" | "flash"
        activation: str = "gelu",
        # In-model patchification: learnable Platonic EdgeConv (FPS centers
        # + KNN + per-edge MLP + max-pool). When off, callers are expected
        # to pre-patchify in the dataloader (StandardPatchifier just lifts +
        # embeds + adds APE). Ratio for FPS = num_centers / num_points.
        edge_conv_patchify: bool = False,
        edge_conv_k: int = 32,
        fps_ratio: float = 0.0625,          # FPS sampling ratio for in-model patchify.
        patchifier: Optional[nn.Module] = None,
    ):
        super().__init__()

        # Resolve activation function from string
        _activations = {"gelu": F.gelu, "silu": F.silu, "relu": F.relu, "mish": F.mish}
        if activation not in _activations:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(_activations.keys())}")
        _activation_fn = _activations[activation]

        if scalar_task_level not in ["node", "graph"]:
            raise ValueError("scalar_task_level must be 'node' or 'graph'.")

        if vector_task_level not in ["node", "graph"]:
            raise ValueError("vector_task_level must be 'node' or 'graph'.")

        # --- Group and Dimension Setup ---
        self.group = PLATONIC_GROUPS[solid_name.lower()]
        self.num_G = self.group.G
        self.hidden_dim = hidden_dim
        self.scalar_task_level = scalar_task_level
        self.vector_task_level = vector_task_level
        self.dense_mode = dense_mode
        self.output_dim = output_dim
        self.output_dim_vec = output_dim_vec
        self.mean_aggregation = mean_aggregation
        self.trivial_readout = trivial_readout
        # When trivial_readout=True the readout MLPs use the trivial-G group
        # (PlatonicLinear with G=1 is just a standard nn.Linear), and the
        # final to_scalars_vectors call reshapes through that trivial group
        # — i.e. the model breaks equivariance at the head.
        if trivial_readout:
            self.readout_group = PLATONIC_GROUPS[f"trivial_{spatial_dim}"]
        else:
            self.readout_group = self.group

        # Patchifier owns x_embedder, APE, and (for EdgeConv) the learnable
        # patchify module. Callers can inject a custom patchifier instance via
        # `patchifier=` (must conform to the interface in patchifiers.py).
        self.edge_conv_patchify = edge_conv_patchify
        if patchifier is not None:
            self.patchifier = patchifier
        elif edge_conv_patchify:
            self.patchifier = EdgeConvPatchifier(
                group=self.group, input_dim=input_dim, input_dim_vec=input_dim_vec,
                hidden_dim=hidden_dim, spatial_dim=spatial_dim, solid_name=solid_name,
                ape_sigma=ape_sigma, learned_freqs=learned_freqs,
                ratio=fps_ratio, k=edge_conv_k, activation=activation,
            )
        else:
            self.patchifier = StandardPatchifier(
                group=self.group, input_dim=input_dim, input_dim_vec=input_dim_vec,
                hidden_dim=hidden_dim, spatial_dim=spatial_dim, solid_name=solid_name,
                ape_sigma=ape_sigma, learned_freqs=learned_freqs, dense_mode=dense_mode,
            )

        # 2. Equivariant Encoder Layers
        # The blocks operate on the total flattened dimension (G * C).
        dim_feedforward = int(self.hidden_dim * ffn_dim_factor)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(PlatonicBlock(
                d_model=self.hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                solid_name=solid_name,
                dropout=dropout,
                activation=_activation_fn,
                drop_path=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
                norm_type=norm_type,
                freq_sigma=rope_sigma,
                freq_init=freq_init,
                learned_freqs=learned_freqs,
                spatial_dims=spatial_dim,
                mean_aggregation=mean_aggregation,
                attention=attention,
                use_key=use_key,
                rope_on_values=rope_on_values,
                rope_v_separate_freqs=rope_v_separate_freqs,
                attention_backend=attention_backend,
            ))

        readout_solid = f"trivial_{spatial_dim}" if trivial_readout else solid_name
        readout_G = 1 if trivial_readout else self.num_G
        if ffn_readout:
            self.scalar_readout = nn.Sequential(
                PlatonicLinear(self.hidden_dim, self.hidden_dim, readout_solid),
                nn.GELU(),
                PlatonicLinear(self.hidden_dim, readout_G * output_dim, readout_solid)
            )
            self.vector_readout = nn.Sequential(
                PlatonicLinear(self.hidden_dim, self.hidden_dim, readout_solid),
                nn.GELU(),
                PlatonicLinear(self.hidden_dim, readout_G * output_dim_vec * spatial_dim, readout_solid)
            )
        else:
            self.scalar_readout = PlatonicLinear(self.hidden_dim, readout_G * output_dim, readout_solid)
            self.vector_readout = PlatonicLinear(self.hidden_dim, readout_G * output_dim_vec * spatial_dim, readout_solid)

    def forward(self,
                x: Tensor,
                pos: Tensor,
                batch: Optional[torch.Tensor] = None,
                mask: Optional[Tensor] = None,
                vec: Optional[Tensor] = None,
                avg_num_nodes: float = 1.0) -> Tensor:
        """
        Args:
            x: Node scalar features `(N, input_dim)` or `None`.
            pos: Node positions `(N, spatial_dims)`.
            batch: Batch index per node `(N,)`. Pass `None` for already-dense input.
            vec: Node vector features `(N, input_dim_vec, spatial_dims)` or `None`.

        Returns:
            `(scalars, vectors)`. Scalars have shape `(B, output_dim)` for graph
            tasks or `(N, output_dim)` for node tasks.
        """
        # 1. Patchifier handles dense conversion (if requested), lifting,
        # input embedding, and APE. Returns a 4-tuple where exactly one of
        # mask/batch is non-None depending on the patchifier's output format.
        x, pos, mask, batch = self.patchifier(x, vec, pos, batch)

        # 2. Equivariant Encoder
        for layer in self.layers:
            x = layer(x=x, pos=pos, batch=batch, mask=mask,
                      avg_num_nodes=avg_num_nodes)

        # 3. Post-pooling readout. The patchifier records whether the *original*
        # input was already dense; that controls how node-task readout
        # extracts per-node features.
        input_was_dense = getattr(self.patchifier, 'input_was_dense_format', False)
        if self.scalar_task_level == "graph":
            scalar_x = pool(x, batch, mask, avg_num_nodes, self.dense_mode, self.mean_aggregation)
        else:
            scalar_x = x[mask] if (not input_was_dense and self.dense_mode) else x

        if self.vector_task_level == "graph":
            vector_x = pool(x, batch, mask, avg_num_nodes, self.dense_mode, self.mean_aggregation)
        else:
            vector_x = x[mask] if (not input_was_dense and self.dense_mode) else x

        scalar_x = self.scalar_readout(scalar_x)
        vector_x = self.vector_readout(vector_x)

        # 4. Extract the scalar and vector parts. With trivial_readout=True
        # the readout group is the trivial G=1 group, so to_scalars_vectors
        # reshapes through that — symmetry is broken at the head.
        scalars = to_scalars_vectors(scalar_x, self.output_dim, 0, self.readout_group)[0]
        vectors = to_scalars_vectors(vector_x, 0, self.output_dim_vec, self.readout_group)[1]
        return scalars, vectors
