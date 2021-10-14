import numpy as np
import torch

from .base_layers import Dense


class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def __init__(self, emb_size, name=None):
        super().__init__()
        self.emb_size = emb_size

        # Atom embeddings: We go up to Pu (94). Use 93 dimensions because of 0-based indexing
        self.embeddings = torch.nn.Embedding(93, emb_size)
        # init by uniform distribution
        torch.nn.init.uniform_(self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, Z):
        """
        Returns
        -------
            h: Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(Z - 1)  # -1 because Z.min()=1 (==Hydrogen)
        return h


class EdgeEmbedding(torch.nn.Module):
    """
    Edge embedding based on the concatenation of atom embeddings and subsequent dense layer.

    Parameters
    ----------
        atom_features: int
            Embedding size of the atom embeddings.
        edge_features: int
            Embedding size of the edge embeddings.
        out_features: int
            Embedding size after the dense layer.
        activation: str
            Activation function used in the dense layer.
    """

    def __init__(
        self, atom_features, edge_features, out_features, activation=None, name=None
    ):
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = Dense(in_features, out_features, activation=activation, bias=False)

    def forward(self, h, m_rbf, idnb_a, idnb_c,):
        """
        Returns
        -------
            m_ca: Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        # m_rbf: shape (nEdges, nFeatures)
        # in embedding block: m_rbf = rbf ; In interaction block: m_rbf = m_ca

        h_a = h[idnb_a]  # shape=(nEdges, emb_size)
        h_c = h[idnb_c]  # shape=(nEdges, emb_size)

        m_ca = torch.cat([h_a, h_c, m_rbf], dim=-1)  # (nEdges, 2*emb_size+nFeatures)
        m_ca = self.dense(m_ca)  # (nEdges, emb_size)
        return m_ca
