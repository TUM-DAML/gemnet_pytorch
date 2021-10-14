import torch
from torch_scatter import scatter
from .base_layers import ResidualLayer, Dense
from ..initializers import he_orthogonal_init
from .scaling import ScalingFactor
from .embedding_block import EdgeEmbedding


class AtomUpdateBlock(torch.nn.Module):
    """
    Aggregate the message embeddings of the atoms

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edge embeddings.
        nHidden: int
            Number of residual blocks.
        activation: callable/str
            Activation function to use in the dense layers.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        activation=None,
        scale_file=None,
        name: str = "atom_update",
    ):
        super().__init__()
        self.name = name
        self.emb_size_edge = emb_size_edge

        self.dense_rbf = Dense(emb_size_rbf, emb_size_edge, activation=None, bias=False)
        self.scale_sum = ScalingFactor(scale_file=scale_file, name=name + "_sum")

        self.layers = self.get_mlp(emb_size_atom, nHidden, activation)

    def get_mlp(self, units, nHidden, activation):
        dense1 = Dense(self.emb_size_edge, units, activation=activation, bias=False)
        res = [
            ResidualLayer(units, nLayers=2, activation=activation)
            for i in range(nHidden)
        ]
        mlp = [dense1] + res
        return torch.nn.ModuleList(mlp)

    def forward(self, h, m, rbf, id_j):
        """
        Returns
        -------
            h: Tensor, shape=(nAtoms, emb_size_atom)
                Atom embedding.
        """
        nAtoms = h.shape[0]

        mlp_rbf = self.dense_rbf(rbf)  # (nEdges, emb_size_edge)
        x = m * mlp_rbf

        x2 = scatter(x, id_j, dim=0, dim_size=nAtoms, reduce="add")  
        x = self.scale_sum(m, x2) # (nAtoms, emb_size_edge)

        for i, layer in enumerate(self.layers):
            x = layer(x)  # (nAtoms, emb_size_atom)
        return x


class OutputBlock(AtomUpdateBlock):
    """
    Combines the atom update block and subsequent final dense layer.

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edge embeddings.
        nHidden: int
            Number of residual blocks.
        num_targets: int
            Number of targets.
        activation: str
            Activation function to use in the dense layers (except for the final dense layer).
        direct_forces: bool
            If true directly predict forces without taking the gradient of the energy potential.
        output_init: str
            Kernel initializer of the final dense layer.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        num_targets: int,
        activation=None,
        direct_forces=True,
        output_init="HeOrthogonal",
        scale_file=None,
        name: str = "output",
        **kwargs,
    ):

        super().__init__(
            name=name,
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=nHidden,
            activation=activation,
            scale_file=scale_file,
            **kwargs,
        )

        assert isinstance(output_init, str)
        self.output_init = output_init
        self.direct_forces = direct_forces
        self.dense_rbf = Dense(emb_size_rbf, emb_size_edge, activation=None, bias=False)

        self.seq_energy = self.layers  # inherited from parent class
        # do not add bias to final layer to enforce that prediction for an atom 
        # without any edge embeddings is zero
        self.out_energy = Dense(emb_size_atom, num_targets, bias=False, activation=None)

        if self.direct_forces:
            self.scale_rbf = ScalingFactor(scale_file=scale_file, name=name + "_had")
            self.seq_forces = self.get_mlp(emb_size_edge, nHidden, activation)
            # no bias in final layer to ensure continuity
            self.out_forces = Dense(
                emb_size_edge, num_targets, bias=False, activation=None
            )

        self.reset_parameters()

    def reset_parameters(self):
        if self.output_init.lower() == "heorthogonal":
            he_orthogonal_init(self.out_energy.weight)
            if self.direct_forces:
                he_orthogonal_init(self.out_forces.weight)
        elif self.output_init.lower() == "zeros":
            torch.nn.init.zeros_(self.out_energy.weight)
            if self.direct_forces:
                torch.nn.init.zeros_(self.out_forces.weight)
        else:
            raise UserWarning(f"Unknown output_init: {self.output_init}")

    def forward(self, h, m, rbf, id_j):
        """
        Returns
        -------
            (E, F): tuple
            - E: Tensor, shape=(nAtoms, num_targets)
            - F: Tensor, shape=(nEdges, num_targets)
            Energy and force prediction
        """
        nAtoms = h.shape[0]

        rbf_mlp = self.dense_rbf(rbf)  # (nEdges, emb_size_edge)
        x = m * rbf_mlp

        # -------------------------------------- Energy Prediction -------------------------------------- #
        x_E = scatter(x, id_j, dim=0, dim_size=nAtoms, reduce="add")  # (nAtoms, emb_size_edge)
        x_E = self.scale_sum(m, x_E)

        for i, layer in enumerate(self.seq_energy):
            x_E = layer(x_E)  # (nAtoms, emb_size_atom)

        x_E = self.out_energy(x_E)  # (nAtoms, num_targets)

        # --------------------------------------- Force Prediction -------------------------------------- #
        if self.direct_forces:

            x_F = self.scale_rbf(m, x)

            for i, layer in enumerate(self.seq_forces):
                x_F = layer(x_F)  # (nEdges, emb_size_edge)

            x_F = self.out_forces(x_F)  # (nEdges, num_targets)
        else:
            x_F = 0
        # ----------------------------------------------------------------------------------------------- #

        return x_E, x_F
