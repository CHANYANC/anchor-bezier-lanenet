# bezier_utils.py
import torch


def sample_bezier(control_points, num_samples=40):
    """
    control_points: (N, L, 4, 2) or (L, 4, 2)
    return: (N, L, num_samples, 2)
    """
    if control_points.dim() == 3:
        control_points = control_points.unsqueeze(0)  # (1, L, 4, 2)

    N, L, _, _ = control_points.shape
    device = control_points.device

    t = torch.linspace(0., 1., num_samples, device=device)  # (T,)
    B0 = (1 - t) ** 3
    B1 = 3 * (1 - t) ** 2 * t
    B2 = 3 * (1 - t) * t ** 2
    B3 = t ** 3
    B = torch.stack([B0, B1, B2, B3], dim=0)  # (4, T)

    B = B.view(1, 1, 4, num_samples, 1)  # broadcast
    P = control_points.unsqueeze(3)      # (N, L, 4, 1, 2)

    pts = (B * P).sum(dim=2)  # (N, L, T, 2)
    return pts
