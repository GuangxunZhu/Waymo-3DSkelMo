import torch
import torch.nn as nn


# class GeodesicLoss(nn.Module):
#     def __init__(self):
#         super(GeodesicLoss, self).__init__()

#     def compute_geodesic_distance(self, m1, m2, epsilon=1e-7):
#         """ Compute the geodesic distance between two rotation matrices.
#         Args:
#             m1, m2: Two rotation matrices with the shape (batch x 3 x 3).
#         Returns:
#             The minimal angular difference between two rotation matrices in radian form [0, pi].
#         """
#         batch = m1.shape[0]
#         m = torch.bmm(m1, m2.permute(0, 2, 1))  # batch*3*3

#         cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
#         # cos = (m.diagonal(dim1=-2, dim2=-1).sum(-1) -1) /2
#         # cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
#         # cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)
#         cos = torch.clamp(cos, -1 + epsilon, 1 - epsilon)
#         theta = torch.acos(cos)

#         return theta

#     def __call__(self, m1, m2, reduction='mean'):
#         loss = self.compute_geodesic_distance(m1, m2)

#         if reduction == 'mean':
#             return loss.mean()
#         elif reduction == 'none':
#             return loss
#         else:
#             raise RuntimeError(f'unsupported reduction: {reduction}')
#             raise RuntimeError(f'unsupported reduction: {reduction}')


class GeodesicLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(GeodesicLoss, self).__init__()
        self.reduction = reduction

    def compute_geodesic_distance(self, m1, m2, epsilon=1e-7):
        """Compute the geodesic distance between two rotation matrices."""
        m = torch.bmm(m1, m2.permute(0, 2, 1))  # (B, 3, 3)
        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.clamp(cos, -1 + epsilon, 1 - epsilon)
        theta = torch.acos(cos)
        return theta  # (B,)

    def forward(self, m1, m2):
        loss = self.compute_geodesic_distance(m1, m2)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        else:
            raise RuntimeError(f'Unsupported reduction mode: {self.reduction}')

