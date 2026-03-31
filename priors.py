from abc import abstractmethod, ABCMeta
import torch
from torch import nn
from pytorch_lightning.utilities import rank_zero_warn


__all__ = ["Atomref"]
# 初始化原子参考能量
# 先验模型框架，其中 BasePrior 是一个抽象基类，定义了先验模型的基本接口。
# Atomref 是一个具体的实现，用于处理原子参考能量。
# Atomref 的核心逻辑是通过 nn.Embedding 层将原子类型 z 映射到参考能量，并将其加到模型的预测值 x 上。



class BasePrior(nn.Module, metaclass=ABCMeta):
    r"""Base class for prior models.
    Derive this class to make custom prior models, which take some arguments and a dataset as input.
    As an example, have a look at the `torchmdnet.priors.Atomref` prior.
    """

    def __init__(self, dataset=None):
        super(BasePrior, self).__init__()

    @abstractmethod
    def get_init_args(self):
        r"""A function that returns all required arguments to construct a prior object.
        The values should be returned inside a dict with the keys being the arguments' names.
        All values should also be saveable in a .yaml file as this is used to reconstruct the
        prior model from a checkpoint file.
        """
        return

    @abstractmethod
    def forward(self, x, z, pos, batch):
        r"""Forward method of the prior model.

        Args:
            x (torch.Tensor): scalar atomwise predictions from the model.
            z (torch.Tensor): atom types of all atoms.
            pos (torch.Tensor): 3D atomic coordinates.
            batch (torch.Tensor): tensor containing the sample index for each atom.

        Returns:
            torch.Tensor: updated scalar atomwise predictions
        """
        return


class Atomref(BasePrior):
    r"""Atomref prior model.
    When using this in combination with some dataset, the dataset class must implement
    the function `get_atomref`, which returns the atomic reference values as a tensor.
    """

    def __init__(self, max_z=None, dataset=None):
        super(Atomref, self).__init__()
        if max_z is None and dataset is None:
            raise ValueError("Can't instantiate Atomref prior, all arguments are None.")
        if dataset is None:
            atomref = torch.zeros(max_z, 1)
        else:
            atomref = dataset.get_atomref()
            if atomref is None:
                rank_zero_warn(
                    "The atomref returned by the dataset is None, defaulting to zeros with max. "
                    "atomic number 99. Maybe atomref is not defined for the current target."
                )
                atomref = torch.zeros(100, 1)

        if atomref.ndim == 1:
            atomref = atomref.view(-1, 1)
        self.register_buffer("initial_atomref", atomref)
        self.atomref = nn.Embedding(len(atomref), 1)
        self.atomref.weight.data.copy_(atomref)

    def reset_parameters(self):
        self.atomref.weight.data.copy_(self.initial_atomref)

    def get_init_args(self):
        return dict(max_z=self.initial_atomref.size(0))

    def forward(self, x, z, pos, batch):
        return x + self.atomref(z)


class LongRangeMotifInteraction(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        # 论文公式10：可学习平衡系数 β ∈ [0,1]
        self.beta = nn.Parameter(torch.tensor(0.5))
        # 约束β在0~1之间，保证稳定性
        self.sigmoid = nn.Sigmoid()

    def forward(self, atomic_scalar, atomic_vector, S, motif_masks):
        """
        输入：
            atomic_scalar: [N, hidden]  原子标量特征
            atomic_vector: [N, 3, hidden] 原子矢量特征
            S: [N, M]  影响矩阵
            motif_masks: [M, N]  基序掩码
        输出：
            enhanced_scalar: [N, hidden]  增强后的原子标量特征
            enhanced_vector: [N, 3, hidden] 增强后的原子矢量特征
        """
        beta = self.sigmoid(self.beta)
        num_motifs = motif_masks.shape[0]

        # 构建基序间影响矩阵 S_motif [M,M]
        S_motif = torch.matmul(S.T, torch.matmul(S, motif_masks.T))
        # 屏蔽自影响（m'≠m）
        S_motif = S_motif * (1.0 - torch.eye(num_motifs, device=S_motif.device))

        # 公式8-9：长程特征加权求和
        # 先从掩码反推：每个原子对应的基序索引
        atom2motif = torch.argmax(motif_masks, dim=0)  # [N]

        # 公式10：残差连接融合长程特征
        enhanced_scalar = atomic_scalar + beta * S[:, atom2motif].sum(dim=-1, keepdim=True)
        enhanced_vector = atomic_vector + beta * S[:, atom2motif].sum(dim=-1, keepdim=True).unsqueeze(1)

        return enhanced_scalar, enhanced_vector
