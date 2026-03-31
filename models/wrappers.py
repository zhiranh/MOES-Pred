from abc import abstractmethod, ABCMeta
from torch import nn



class BaseWrapper(nn.Module, metaclass=ABCMeta):
    r"""Base class for model wrappers.

    Children of this class should implement the `forward` method,
    which calls `self.model(z, pos, batch=batch)` at some point.
    Wrappers that are applied before the REDUCE operation should return
    the model's output, `z`, `pos`, `batch` and potentially vector
    features`v`. Wrappers that are applied after REDUCE should only
    return the model's output.
    """

    def __init__(self, model):
        super(BaseWrapper, self).__init__()
        self.model = model

    def reset_parameters(self):
        self.model.reset_parameters()

    # ====================== MOES-Pred 修改：扩展输入输出 ======================
    @abstractmethod
    def forward(self, z, pos, batch=None, motif_feat=None, motif_mask=None, influence_matrix=None):
        return



class AtomFilter(BaseWrapper):
    def __init__(self, model, remove_threshold):
        """
        Wrapper module that filters atoms based on a threshold value.

        Args:
            model (nn.Module): PyTorch model to wrap and filter atoms.
            remove_threshold (float): Threshold value for filtering atoms based on their properties.

        Inputs:
            z (torch.Tensor): Atom properties tensor.
            pos (torch.Tensor): Atom positions tensor.
            batch (torch.Tensor, optional): Batch tensor for grouping atoms. Defaults to None.
            motif_feat (torch.Tensor, optional): BRICS group features for MOES-Pred.
            motif_mask (torch.Tensor, optional): Mask for BRICS groups.
            influence_matrix (torch.Tensor, optional): Interaction matrix between groups.

        Returns:
            tuple: 包含主干模型所有输出 + MOES 特征
        """
        super(AtomFilter, self).__init__(model)
        self.remove_threshold = remove_threshold


    def forward(self, z, pos, batch=None, motif_feat=None, motif_mask=None, influence_matrix=None):

        x, v, z, pos, batch, motif_pred, ens_loss = self.model(
            z, pos, batch=batch,
            motif_feat=motif_feat,
            motif_mask=motif_mask,
            influence_matrix=influence_matrix
        )

        n_samples = len(batch.unique())


        atom_mask = z > self.remove_threshold


        x = x[atom_mask]
        if v is not None:
            v = v[atom_mask]
        z = z[atom_mask]
        pos = pos[atom_mask]
        batch = batch[atom_mask]


        if motif_feat is not None:
            motif_feat = motif_feat[atom_mask]
        if motif_mask is not None:
            motif_mask = motif_mask[atom_mask]
        if motif_pred is not None:
            motif_pred = motif_pred[atom_mask]


        assert len(batch.unique()) == n_samples, (
            "Some samples were completely filtered out by the atom filter. "
            f"Make sure that at least one atom per sample exists with Z > {self.remove_threshold}."
        )


        return x, v, z, pos, batch, motif_pred, ens_loss