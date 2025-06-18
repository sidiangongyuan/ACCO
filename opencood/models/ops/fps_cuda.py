
import torch
from torch.autograd.function import Function, once_differentiable
import sys
sys.path.append(r'/home/yangk/coSparse4D/coSparse4D/opencood/models/ops')

import sample_ext


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, pts: torch.Tensor, K: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set
        """
        assert pts.is_contiguous()

        b, N, _ = pts.size()
        # output_anchor = torch.cuda.IntTensor(B, npoint)
        idx =  torch.cuda.IntTensor(b,K)
        temp = torch.cuda.FloatTensor(b,N).fill_(1e10)
        sample_ext.furthest_point_sampling_wrapper(b, N, K, pts, temp, idx)
        return idx

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply