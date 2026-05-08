from typing import Optional
import torch
import numpy as np
from . import Simulator

class SwissRollSimulator_3d(Simulator):
    """
    Swiss Roll simulator for generating 3D point clouds.
    Each point is independently sampled from either the background or the spiral.
    
    Parameters theta = [amplitude, scale]
    - amplitude: depth of the noise (radial thickness)
    - scale: how fast the spiral expands
    """

    def __init__(
        self,
        ncirc: int,
        nback: int,
        side: float,
        height: float,
        device: str = "cpu",
    ):
        super().__init__()
        self.ncirc = ncirc
        self.nback = nback
        self.side = side
        self.height = height
        self.ntot = ncirc + nback
        self.p = nback / self.ntot  # Probabilità di successo per il campionamento Bernoulli
        self.device = device

    def generate(
        self,
        theta: torch.Tensor,
        n_samples: int,
        seed: Optional[int] = None,
        seed_start: Optional[int] = None,
        desc: Optional[str] = None,
    ) -> torch.Tensor:
        if theta.numel() != 2:
            raise ValueError(f"Expected 2 parameters [amplitude, scale], got {theta.numel()}")

        if seed_start is None:
            seed_start = 0 if seed is None else seed

        result = super().generate(theta, n_samples, seed_start=seed_start, desc=desc)
        return result.to(self.device)

    def generate_single(self, theta: torch.Tensor, seed: int):
        amplitude = float(theta[0])
        scale = float(theta[1])

        generator = torch.Generator()
        generator.manual_seed(seed)

        # 1. Decidi la provenienza di ogni punto (Bernoulli process)
        # 1 se background, 0 se spirale
        is_back = torch.rand(self.ntot, generator=generator) < self.p
        n_back_actual = int(is_back.sum())
        n_circ_actual = self.ntot - n_back_actual

        # Parametri geometrici
        t_min = 0.0
        t_max = 4 * np.pi 
        side = self.side
        height = self.height

        # 2. Generazione BACKGROUND (Punti nel parallelepipedo)
        x_back = (torch.rand(n_back_actual, generator=generator) - 0.5) * 2 * side
        y_back = (torch.rand(n_back_actual, generator=generator) - 0.5) * 2 * side
        z_back = (torch.rand(n_back_actual, generator=generator) - 0.5) * 2 * height
        points_back = torch.stack([x_back, y_back, z_back], dim=-1)

        # 3. Generazione SWISS-ROLL (Punti sulla spirale estesa in Z)
        t = torch.rand(n_circ_actual, generator=generator) * (t_max - t_min) + t_min
        
        # Raggio base + rumore radiale
        base_r = scale * t
        radial_noise = torch.randn(n_circ_actual, generator=generator) * amplitude
        radii = base_r + radial_noise

        x_spiral = radii * torch.cos(t)
        y_spiral = radii * torch.sin(t)
        # La coordinata Z della spirale segue lo stesso range del background
        z_spiral = (torch.rand(n_circ_actual, generator=generator) - 0.5) * 2 * height
        
        points_spiral = torch.stack([x_spiral, y_spiral, z_spiral], dim=-1)

        # 4. Combine e Shuffle
        points = torch.cat([points_back, points_spiral], dim=0)
        
        # Mischiamo i punti affinché l'ordine non rifletta la provenienza
        perm = torch.randperm(self.ntot, generator=generator)
        points = points[perm]

        return points.numpy()
    
    def theoretical_fisher_matrix(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Fisher Teorica 2D Esatta (Coerenza di Misura dx dy).
        Risolve l'incompatibilità tra background cartesiano e spirale polare.
        """
        A = float(theta[0]) 
        S = float(theta[1])
        phi_max = 4 * np.pi
        side = self.side 
        area_quad = (2 * side)**2
        eps = 1e-18

        # 1. Griglia di Integrazione
        r_max_int = max(S * phi_max + 8 * A, np.sqrt(2) * side)
        n_r, n_phi = 5000, 5000
        r_lin = torch.linspace(1e-6, r_max_int, n_r) # Evitiamo r=0 per la divisione 1/R
        phi_lin = torch.linspace(0.0, phi_max, n_phi)
        dr, dphi = r_lin[1] - r_lin[0], phi_lin[1] - phi_lin[0]

        PHI, R = torch.meshgrid(phi_lin, r_lin, indexing='ij')

        # 2. Densità Background (rispetto a dx dy)
        # È costante nel quadrato: 1 / Area
        r_limit_quad = side / torch.max(torch.abs(torch.cos(PHI)), torch.abs(torch.sin(PHI)))
        p_back_xy = torch.zeros_like(R)
        p_back_xy[R <= r_limit_quad] = 1.0 / area_quad

        # 3. Densità Spirale (rispetto a dx dy)
        # p_spiral(x,y) = p_spiral(r,phi) / r
        z = (R - S * PHI) / A
        phi_std = torch.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        
        # Qui applichiamo il fattore 1/R per la coerenza di misura
        p_spiral_xy = ((1.0 / phi_max) * (phi_std / A)) / R

        # 4. Mistura Totale (tutto in dx dy)
        prob_total_xy = (self.p * p_back_xy) + ((1 - self.p) * p_spiral_xy)

        # 5. Gradienti (rispetto a dx dy)
        # Deriviamo p_spiral_xy rispetto ai parametri
        dp_dA_xy = (1 - self.p) * p_spiral_xy * (z**2 - 1) / A
        dp_dS_xy = (1 - self.p) * p_spiral_xy * (z * PHI) / A

        # 6. Fisher Matrix
        # Integriamo (Grad^2 / Prob) * r * dr * dphi
        # Il termine R è lo Jacobiano dell'integrazione dx dy -> r dr dphi
        score_A2 = (dp_dA_xy**2) / (prob_total_xy + eps)
        score_S2 = (dp_dS_xy**2) / (prob_total_xy + eps)
        score_AS = (dp_dA_xy * dp_dS_xy) / (prob_total_xy + eps)

        # Integrazione numerica finale
        f_AA = torch.sum(score_A2 * R) * dr * dphi
        f_SS = torch.sum(score_S2 * R) * dr * dphi
        f_AS = torch.sum(score_AS * R) * dr * dphi

        fisher_matrix = torch.tensor([[f_AA, f_AS], [f_AS, f_SS]]) * self.ntot
        return fisher_matrix.float()

    def sorted_distance_summary(self, point_clouds):
        if isinstance(point_clouds, torch.Tensor):
            point_clouds = [point_clouds[i] for i in range(point_clouds.shape[0])]
        sorted_dists = []
        for pts in point_clouds:
            if not isinstance(pts, torch.Tensor):
                pts = torch.tensor(pts, dtype=torch.float32)
            dists = torch.norm(pts, dim=-1)
            sorted_dists.append(torch.sort(dists)[0])
        return sorted_dists

    def mean_distance_summary(self, point_clouds):
        if isinstance(point_clouds, torch.Tensor):
            point_clouds = [point_clouds[i] for i in range(point_clouds.shape[0])]
        summaries = []
        for pts in point_clouds:
            if not isinstance(pts, torch.Tensor):
                pts = torch.tensor(pts, dtype=torch.float32)
            dists = torch.norm(pts, dim=-1)
            summaries.append(torch.stack([torch.mean(dists), torch.std(dists)]))
        return summaries