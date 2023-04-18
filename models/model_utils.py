import torch


def cal_distance(m: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    vector = m - n
    distance = torch.norm(vector, p=2, dim=1)
    torch.clamp_(distance, min=0.1)
    return distance


def cal_angle(a, b, c):
    ba = a - b
    bc = c - b

    dot = torch.matmul(ba.unsqueeze(-1).transpose(-2, -1), bc.unsqueeze(-1))
    cosine_angle = dot.squeeze(-1) / (torch.norm(ba, p=2, dim=1).reshape(-1, 1) * torch.norm(bc, p=2, dim=1).reshape(-1, 1))
    cosine_angle = torch.where(torch.logical_or(cosine_angle > 1, cosine_angle < -1), torch.round(cosine_angle), cosine_angle)
    angle = torch.arccos(cosine_angle)

    return angle


def cal_dihedral(a, b, c, d):
    ab = a - b
    cb = c - b
    dc = d - c

    cb /= torch.norm(cb, p=2, dim=1).reshape(-1, 1)
    v = ab - torch.matmul(ab.unsqueeze(-1).transpose(-2, -1), cb.unsqueeze(-1)).squeeze(-1) * cb
    w = dc - torch.matmul(dc.unsqueeze(-1).transpose(-2, -1), cb.unsqueeze(-1)).squeeze(-1) * cb
    x = torch.matmul(v.unsqueeze(-1).transpose(-2, -1), w.unsqueeze(-1)).squeeze(-1)
    y = torch.matmul(torch.cross(cb, v).unsqueeze(-1).transpose(-2, -1), w.unsqueeze(-1)).squeeze(-1)

    return torch.atan2(y, x)