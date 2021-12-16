import numpy as np
import torch
import sympy as sym

from .envelope import Envelope
from .basis_utils import bessel_basis, real_sph_harm


class BesselBasisLayer(torch.nn.Module):
    """
    1D Bessel Basis

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    envelope_exponent: int = 5
        Exponent of the envelope function.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int = 5,
        name="bessel_basis",
    ):
        super().__init__()
        self.num_radial = num_radial
        self.inv_cutoff = 1 / cutoff
        self.norm_const = (2 * self.inv_cutoff) ** 0.5

        self.envelope = Envelope(envelope_exponent)

        # Initialize frequencies at canonical positions
        self.frequencies = torch.nn.Parameter(
            data=torch.Tensor(
                np.pi * np.arange(1, self.num_radial + 1, dtype=np.float32)
            ),
            requires_grad=True,
        )

    def forward(self, d):
        d = d[:, None]  # (nEdges,1)
        d_scaled = d * self.inv_cutoff
        env = self.envelope(d_scaled)
        return env * self.norm_const * torch.sin(self.frequencies * d_scaled) / d


class SphericalBasisLayer(torch.nn.Module):
    """
    2D Fourier Bessel Basis

    Parameters
    ----------
    num_spherical: int
        Controls maximum frequency.
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    envelope_exponent: int = 5
        Exponent of the envelope function.
    efficient: bool
        Whether to use the (memory) efficient implementation or not.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int = 5,
        efficient: bool = False,
        name: str = "spherical_basis",
    ):
        super().__init__()

        assert num_radial <= 64
        self.efficient = efficient
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.envelope = Envelope(envelope_exponent)
        self.inv_cutoff = 1 / cutoff

        # retrieve formulas
        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(
            num_spherical, spherical_coordinates=True, zero_m_only=True
        )
        self.sph_funcs = []  # (num_spherical,)
        self.bessel_funcs = []  # (num_spherical * num_radial,)
        self.norm_const = self.inv_cutoff ** 1.5
        self.register_buffer(
            "device_buffer", torch.zeros(0), persistent=False
        )  # dummy buffer to get device of layer

        # convert to torch functions
        x = sym.symbols("x")
        theta = sym.symbols("theta")
        modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
        m = 0  # only single angle
        for l in range(len(Y_lm)):  # num_spherical
            if l == 0: 
                # Y_00 is only a constant -> function returns value and not tensor
                first_sph = sym.lambdify([theta], Y_lm[l][m], modules)
                self.sph_funcs.append(
                    lambda theta: torch.zeros_like(theta) + first_sph(theta)
                )
            else:
                self.sph_funcs.append(sym.lambdify([theta], Y_lm[l][m], modules))
            for n in range(num_radial):
                self.bessel_funcs.append(
                    sym.lambdify([x], bessel_formulas[l][n], modules)
                )

    def forward(self, D_ca, Angle_cab, id3_reduce_ca, Kidx):

        d_scaled = D_ca * self.inv_cutoff  # (nEdges,)
        u_d = self.envelope(d_scaled)
        rbf = [f(d_scaled) for f in self.bessel_funcs]
        # s: 0 0 0 0 1 1 1 1 ...
        # r: 0 1 2 3 0 1 2 3 ...
        rbf = torch.stack(rbf, dim=1)  # (nEdges, num_spherical * num_radial)
        rbf = rbf * self.norm_const
        rbf_env = u_d[:, None] * rbf  # (nEdges, num_spherical * num_radial)

        sph = [f(Angle_cab) for f in self.sph_funcs]
        sph = torch.stack(sph, dim=1)  # (nTriplets, num_spherical)

        if not self.efficient:
            rbf_env = rbf_env[id3_reduce_ca]  # (nTriplets, num_spherical * num_radial)
            rbf_env = rbf_env.view(-1, self.num_spherical, self.num_radial)
            # e.g. num_spherical = 3, num_radial = 2
            # z_ln: l: 0 0  1 1  2 2
            #       n: 0 1  0 1  0 1
            sph = sph.view(-1, self.num_spherical, 1)  # (nTriplets, num_spherical, 1)
            # e.g. num_spherical = 3, num_radial = 2
            # Y_lm: l: 0 0  1 1  2 2
            #       m: 0 0  0 0  0 0
            out = (rbf_env * sph).view(-1, self.num_spherical * self.num_radial)
            return out  # (nTriplets, num_spherical * num_radial)
        else:
            rbf_env = rbf_env.view(-1, self.num_spherical, self.num_radial)
            rbf_env = torch.transpose(
                rbf_env, 0, 1
            )  # (num_spherical, nEdges, num_radial)

            # Zero padded dense matrix
            # maximum number of neighbors, catch empty id_reduce_ji with maximum
            Kmax = 0 if sph.shape[0]==0 else torch.max(torch.max(Kidx + 1), torch.tensor(0))  
            nEdges = d_scaled.shape[0]

            sph2 = torch.zeros(
                nEdges, Kmax, self.num_spherical, device=self.device_buffer.device, dtype=sph.dtype
            )
            sph2[id3_reduce_ca, Kidx] = sph

            # (num_spherical, nEdges, num_radial), (nEdges, Kmax, num_spherical)
            return rbf_env, sph2


class TensorBasisLayer(torch.nn.Module):
    """
    3D Fourier Bessel Basis

    Parameters
    ----------
    num_spherical: int
        Controls maximum frequency.
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    envelope_exponent: int = 5
        Exponent of the envelope function.
    efficient: bool
        Whether to use the (memory) efficient implementation or not.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int = 5,
        efficient=False,
        name: str = "tensor_basis",
    ):
        super().__init__()

        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.efficient = efficient

        self.inv_cutoff = 1 / cutoff
        self.envelope = Envelope(envelope_exponent)

        # retrieve formulas
        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(
            num_spherical, spherical_coordinates=True, zero_m_only=False
        )
        self.sph_funcs = []  # (num_spherical**2,)
        self.bessel_funcs = []  # (num_spherical * num_radial,)
        self.norm_const = self.inv_cutoff ** 1.5

        # convert to tensorflow functions
        x = sym.symbols("x")
        theta = sym.symbols("theta")
        phi = sym.symbols("phi")
        modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
        for l in range(len(Y_lm)):  # num_spherical
            for m in range(len(Y_lm[l])):
                if (
                    l == 0
                ):  # Y_00 is only a constant -> function returns value and not tensor
                    first_sph = sym.lambdify([theta, phi], Y_lm[l][m], modules)
                    self.sph_funcs.append(
                        lambda theta, phi: torch.zeros_like(theta)
                        + first_sph(theta, phi)
                    )
                else:
                    self.sph_funcs.append(
                        sym.lambdify([theta, phi], Y_lm[l][m], modules)
                    )
            for j in range(num_radial):
                self.bessel_funcs.append(
                    sym.lambdify([x], bessel_formulas[l][j], modules)
                )

        self.register_buffer(
            "degreeInOrder", torch.arange(num_spherical) * 2 + 1, persistent=False
        ) 

    def forward(self, D_ca, Alpha_cab, Theta_cabd, id4_reduce_ca, Kidx):

        d_scaled = D_ca * self.inv_cutoff
        u_d = self.envelope(d_scaled)

        rbf = [f(d_scaled) for f in self.bessel_funcs]
        # s: 0 0 0 0 1 1 1 1 ...
        # r: 0 1 2 3 0 1 2 3 ...
        rbf = torch.stack(rbf, dim=1)  # (nEdges, num_spherical * num_radial)
        rbf = rbf * self.norm_const

        rbf_env = u_d[:, None] * rbf  # (nEdges, num_spherical * num_radial)
        rbf_env = rbf_env.view(
            (-1, self.num_spherical, self.num_radial)
        )  # (nEdges, num_spherical, num_radial)
        rbf_env = torch.repeat_interleave(
            rbf_env, self.degreeInOrder, dim=1
        )  # (nEdges, num_spherical**2, num_radial)

        if not self.efficient:
            rbf_env = rbf_env.view(
                (-1, self.num_spherical ** 2 * self.num_radial)
            )  # (nEdges, num_spherical**2 * num_radial)
            rbf_env = rbf_env[
                id4_reduce_ca
            ]  # (nQuadruplets, num_spherical**2 * num_radial)
            # e.g. num_spherical = 3, num_radial = 2
            # j_ln: l: 0  0    1  1  1  1  1  1    2  2  2  2  2  2  2  2  2  2
            #       n: 0  1    0  1  0  1  0  1    0  1  0  1  0  1  0  1  0  1

        sph = [f(Alpha_cab, Theta_cabd) for f in self.sph_funcs]
        sph = torch.stack(sph, dim=1)  # (nQuadruplets, num_spherical**2)

        if not self.efficient:
            sph = torch.repeat_interleave(
                sph, self.num_radial, axis=1
            )  # (nQuadruplets, num_spherical**2 * num_radial)
            # e.g. num_spherical = 3, num_radial = 2
            # Y_lm: l: 0  0    1  1  1  1  1  1    2  2  2  2  2  2  2  2  2  2
            #       m: 0  0   -1 -1  0  0  1  1   -2 -2 -1 -1  0  0  1  1  2  2
            return rbf_env * sph  # (nQuadruplets, num_spherical**2 * num_radial)

        else:
            rbf_env = torch.transpose(rbf_env, 0, 1)  # (num_spherical**2, nEdges, num_radial)

            # Zero padded dense matrix
            # maximum number of neighbors, catch empty id_reduce_ji with maximum
            Kmax = 0 if sph.shape[0]==0 else torch.max(torch.max(Kidx + 1), torch.tensor(0))  
            nEdges = d_scaled.shape[0]

            sph2 = torch.zeros(
                nEdges, Kmax, self.num_spherical ** 2, device=self.degreeInOrder.device, dtype=sph.dtype
            )
            sph2[id4_reduce_ca, Kidx] = sph

            # (num_spherical**2, nEdges, num_radial), (nEdges, Kmax, num_spherical**2)
            return rbf_env, sph2
