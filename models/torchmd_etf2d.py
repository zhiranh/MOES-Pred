from typing import Optional, Tuple
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot_orthogonal
from torch_scatter import scatter
from torchmdnet.models.utils import (
    NeighborEmbedding,
    CosineCutoff,
    Distance,
    rbf_class_mapping,
    act_class_mapping,
)

from torch.nn.parameter import Parameter
from torch.nn import Linear

from torchmdnet.models.feats import dist_emb, angle_emb, torsion_emb, xyz_to_dat
import torch.nn.functional as F



class EnergySentinel(nn.Module):
    def __init__(self, lambda_penalty=1.0):
        super().__init__()
        self.lambda_penalty = lambda_penalty
        self.rec_loss_fn = nn.MSELoss(reduction='none')

    def score_noise_scheme(self, pred_pos, true_pos, atomic_weights, potential_energy):
        rec_loss_per_atom = self.rec_loss_fn(pred_pos, true_pos).sum(dim=-1)
        rec_loss = (atomic_weights * rec_loss_per_atom).mean()
        aux_loss = self.lambda_penalty * torch.abs(potential_energy)
        score = rec_loss + aux_loss
        return score, rec_loss, aux_loss


class MolecularPotential(nn.Module):
    def __init__(self, cutoff=5.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, pos):
        dist = torch.cdist(pos, pos)
        dist = dist + torch.eye(dist.size(0), device=dist.device) * 1e6
        lj = (1.0 / dist) ** 12 - (1.0 / dist) ** 6
        lj = torch.where(dist < self.cutoff, lj, torch.zeros_like(lj))
        return lj.sum(dim=-1)


class LongRangeMotifInteraction(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.sigmoid = nn.Sigmoid()

    def forward(self, atomic_scalar, atomic_vector, S, motif_masks):
        beta = self.sigmoid(self.beta)
        num_motifs = motif_masks.shape[0]
        S_motif = torch.matmul(S.T, torch.matmul(S, motif_masks.T))
        S_motif = S_motif * (1.0 - torch.eye(num_motifs, device=S_motif.device))
        atom2motif = torch.argmax(motif_masks, dim=0)
        weight = beta * S[:, atom2motif].sum(dim=-1, keepdim=True)
        enhanced_scalar = atomic_scalar + weight * atomic_scalar
        enhanced_vector = atomic_vector + weight.unsqueeze(1) * atomic_vector
        return enhanced_scalar, enhanced_vector


class InfluenceMatrix(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gnn_repr, motif_masks):
        num_atoms = gnn_repr.shape[0]
        h = gnn_repr.unsqueeze(0).repeat(num_atoms, 1, 1)
        h_neg_u = h.clone()
        h_neg_u[range(num_atoms), range(num_atoms)] = 0.0
        s_uv = torch.norm(h - h_neg_u, p=2, dim=-1)
        S = torch.matmul(s_uv, motif_masks.T)
        motif_size = motif_masks.sum(dim=-1) + 1e-8
        S = S / motif_size.unsqueeze(0)
        atomic_weights = s_uv.mean(dim=1)
        motif_weights = torch.matmul(motif_masks, atomic_weights)
        return S, atomic_weights, motif_weights, s_uv


class BRICSMotifAggregation(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

    def forward(self, atomic_scalar, atomic_vector, motif_masks):
        eps = 1e-8
        motif_size = motif_masks.sum(dim=-1, keepdim=True) + eps
        motif_scalar = torch.matmul(motif_masks, atomic_scalar)
        motif_scalar = motif_scalar / motif_size
        motif_vector = torch.einsum("mn, ndc -> mdc", motif_masks, atomic_vector)
        motif_vector = motif_vector / motif_size.unsqueeze(-1)
        return motif_scalar, motif_vector


# ======================================================================================

class EdgeFeatureInit(nn.Module):
    def __init__(self, distance_exp, activation, num_radial, hidden_channels) -> None:
        super().__init__()
        self.distance_exp = distance_exp
        self.act = activation()
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, node_embs, edge_index, edge_weight):
        rbf = self.distance_exp(edge_weight)
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(torch.cat([node_embs[edge_index[0]], node_embs[edge_index[1]], rbf0], dim=-1)))
        e2 = self.lin_rbf_1(rbf) * e1
        return e1, e2


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act):
        super(ResidualLayer, self).__init__()
        self.act = act()
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class UpdateE(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion,
                 num_spherical, num_radial,
                 num_before_skip, num_after_skip, act):
        super(UpdateE, self).__init__()
        self.act = act()
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size_angle, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size_angle, int_emb_size, bias=False)
        self.lin_t1 = nn.Linear(num_spherical * num_spherical * num_radial, basis_emb_size_torsion, bias=False)
        self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)])

        self.e0_norm = nn.LayerNorm(hidden_channels)
        self.e1_norm = nn.LayerNorm(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_t1.weight, scale=2.0)
        glorot_orthogonal(self.lin_t2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(self, x, emb, idx_kj, idx_ji):
        rbf0, sbf, t = emb
        x1, _ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        t = self.lin_t1(t)
        t = self.lin_t2(t)
        x_kj = x_kj * t

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1

        e1 = self.e0_norm(e1)
        e2 = self.e1_norm(e2)
        return e1, e2


def check_for_nan(module, gin, gout):
    if gin[0].isnan().any():
        print("NaN values found in gradients!")


class EMB(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super(EMB, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.torsion_emb = torsion_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.reset_parameters()

    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def forward(self, dist, angle, torsion, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        torsion_emb = self.torsion_emb(dist, angle, torsion, idx_kj)
        return dist_emb, angle_emb, torsion_emb


class TorchMD_ETF2D(nn.Module):
    def __init__(
            self,
            hidden_channels=128,
            num_layers=6,
            num_rbf=50,
            rbf_type="expnorm",
            trainable_rbf=True,
            activation="silu",
            attn_activation="silu",
            neighbor_embedding=True,
            num_heads=8,
            distance_influence="both",
            cutoff_lower=0.0,
            cutoff_upper=5.0,
            max_z=100,
            max_num_neighbors=32,
            layernorm_on_vec=None,
            md17=False,
            seperate_noise=False,
            num_spherical=3, num_radial=6, envelope_exponent=5,
            int_emb_size=64,
            basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8,
            num_before_skip=1, num_after_skip=2
    ):
        super(TorchMD_ETF2D, self).__init__()

        assert distance_influence in ["keys", "values", "both", "none"]
        assert rbf_type in rbf_class_mapping
        assert activation in act_class_mapping
        assert attn_activation in act_class_mapping

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z
        self.layernorm_on_vec = layernorm_on_vec

        act_class = act_class_mapping[activation]

        if self.max_z > 200:
            max_z = self.max_z // 2
            self.embedding = nn.Embedding(max_z, hidden_channels // 2)
            self.type_embedding = nn.Embedding(2, hidden_channels // 2)
        else:
            self.embedding = nn.Embedding(self.max_z, hidden_channels)

        self.distance = Distance(
            cutoff_lower,
            cutoff_upper,
            max_num_neighbors=max_num_neighbors,
            return_vecs=True,
            loop=True,
        )
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = NeighborEmbedding(
            hidden_channels, num_rbf, cutoff_lower, cutoff_upper, self.max_z
        ).jittable() if neighbor_embedding else None

        self.attention_layers = nn.ModuleList()
        self.md17 = md17
        if not self.md17:
            self.vec_norms = nn.ModuleList()
            self.x_norms = nn.ModuleList()

        for _ in range(num_layers):
            layer = EquivariantMultiHeadAttention(
                hidden_channels,
                num_rbf,
                distance_influence,
                num_heads,
                act_class,
                attn_activation,
                cutoff_lower,
                cutoff_upper,
            ).jittable()
            self.attention_layers.append(layer)
            if not self.md17:
                self.vec_norms.append(EquivariantLayerNorm(hidden_channels))
                self.x_norms.append(nn.LayerNorm(hidden_channels))

        self.out_norm = nn.LayerNorm(hidden_channels)
        self.seperate_noise = seperate_noise
        if self.seperate_noise:
            self.out_norm_vec = EquivariantLayerNorm(hidden_channels)

        self.init_e = EdgeFeatureInit(self.distance_expansion, act_class, num_rbf, hidden_channels)
        self.update_es = torch.nn.ModuleList([
            UpdateE(hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion,
                    num_spherical, num_radial, num_before_skip, num_after_skip, act_class) for _ in range(num_layers)])
        self.emb = EMB(num_spherical, num_radial, cutoff_upper, envelope_exponent)

        # ==================== =====================
        self.motif_agg = BRICSMotifAggregation(hidden_channels)
        self.influence_matrix = InfluenceMatrix()
        self.long_range_motif = LongRangeMotifInteraction(hidden_channels)
        self.energy_sentinel = EnergySentinel(lambda_penalty=1.0)
        self.potential_energy = MolecularPotential(cutoff=cutoff_upper)
        self.motif_predictor = nn.Linear(hidden_channels, hidden_channels)
        # ==================================================================================

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()
        if self.layernorm_on_vec:
            self.out_norm_vec.reset_parameters()
        self.motif_predictor.reset_parameters()

    # =========================================
    def forward(self, z, pos, batch, return_e=False, type_idx=None, motif_masks=None, pos_gt=None):

        x = self.embedding(z)
        if type_idx is not None and hasattr(self, 'type_embedding'):
            type_emb = self.type_embedding(type_idx)
            x = torch.concat([x, type_emb], dim=1)


        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)
        x_enhanced, vec_enhanced = x.clone(), vec.clone()

        motif_pred = torch.zeros(1, self.hidden_channels, device=x.device)
        atomic_weights = torch.ones(x.size(0), device=x.device)
        sk_score = torch.tensor(0.0, device=x.device)
        ens_loss = torch.tensor(0.0, device=x.device)


        if motif_masks is not None:
            motif_scalar, motif_vector = self.motif_agg(x, vec, motif_masks)
            motif_pred = self.motif_predictor(motif_scalar)
            S, atomic_weights, motif_weights, s_uv = self.influence_matrix(x, motif_masks)
            x_enhanced, vec_enhanced = self.long_range_motif(x, vec, S, motif_masks)

        x, vec = x_enhanced, vec_enhanced


        num_nodes = z.size(0)
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        e = self.init_e(x, edge_index, edge_weight)
        mask = edge_index[0] != edge_index[1]
        no_loop_edge_index = edge_index[:, mask]


        dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(pos, no_loop_edge_index, num_nodes, use_torsion=True)
        emb = self.emb(dist, angle, torsion, idx_kj)
        edge_attr = self.distance_expansion(edge_weight)


        mask_loop = edge_index[0] != edge_index[1]
        edge_vec[mask_loop] = F.normalize(edge_vec[mask_loop], dim=1)


        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

        update_e0, update_e1 = e[0][mask], e[1][mask]


        for lidx, attn in enumerate(self.attention_layers):
            update_e0, update_e1 = self.update_es[lidx]((update_e0, update_e1), emb, idx_kj, idx_ji)
            edge_e1 = e[1].clone()
            edge_e1[mask] = update_e1


            edge_e1 = edge_e1 * atomic_weights[edge_index[0]].unsqueeze(-1)

            dx, dvec = attn(x, vec, edge_index, edge_weight, edge_e1, edge_vec)
            x = x + dx
            vec = vec + dvec
            if not self.md17:
                vec = self.vec_norms[lidx](vec)


        xnew = self.out_norm(x)
        if self.layernorm_on_vec and hasattr(self, 'out_norm_vec'):
            vec = self.out_norm_vec(vec)


        if pos_gt is not None:
            atom_pot = self.potential_energy(pos)
            weighted_energy = (atomic_weights * atom_pot).sum()
            sk_score, rec_loss, aux_loss = self.energy_sentinel.score_noise_scheme(
                pred_pos=pos, true_pos=pos_gt, atomic_weights=atomic_weights, potential_energy=weighted_energy
            )
            ens_loss = rec_loss + aux_loss


        noise_pred = vec.mean(dim=1) if self.seperate_noise else vec.sum(dim=1)


        deriv = None
        if self.training:
            pos.requires_grad_(True)
            deriv = torch.autograd.grad(
                xnew.sum(), pos,
                create_graph=True, retain_graph=True, allow_unused=True
            )[0]


        return xnew, noise_pred, deriv, motif_pred, ens_loss, sk_score

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers})"
        )


class EquivariantMultiHeadAttention(MessagePassing):
    def __init__(
            self,
            hidden_channels,
            num_rbf,
            distance_influence,
            num_heads,
            activation,
            attn_activation,
            cutoff_lower,
            cutoff_upper,
    ):
        super(EquivariantMultiHeadAttention, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)

        self.dk_proj = None
        if distance_influence in ["keys", "both"]:
            self.dk_proj = nn.Linear(hidden_channels, hidden_channels)

        self.dv_proj = None
        if distance_influence in ["values", "both"]:
            self.dv_proj = nn.Linear(hidden_channels, hidden_channels * 3)

        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        x = self.layernorm(x)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim * 3)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        dk = self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads,
                                                  self.head_dim) if self.dk_proj is not None else None
        dv = self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads,
                                                  self.head_dim * 3) if self.dv_proj is not None else None

        x, vec = self.propagate(edge_index, q=q, k=k, v=v, vec=vec, dk=dk, dv=dv, r_ij=r_ij, d_ij=d_ij, size=None)
        x = x.reshape(-1, self.hidden_channels)
        vec = vec.reshape(-1, 3, self.hidden_channels)

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec
        return dx, dvec

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:
            attn = (q_i * k_j * dk).sum(dim=-1)

        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        if dv is not None:
            v_j = v_j * dv
        x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=2)

        x = x * F.softmax(attn, dim=-1).unsqueeze(2)
        vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(2).unsqueeze(3)
        return x, vec

    def aggregate(self, features: Tuple[torch.Tensor, torch.Tensor], index: torch.Tensor, ptr: Optional[torch.Tensor],
                  dim_size: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
        return x, vec

    def update(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class EquivariantLayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "elementwise_linear"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_linear: bool

    def __init__(
            self,
            normalized_shape: int,
            eps: float = 1e-5,
            elementwise_linear: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(EquivariantLayerNorm, self).__init__()

        self.normalized_shape = (int(normalized_shape),)
        self.eps = eps
        self.elementwise_linear = elementwise_linear
        if self.elementwise_linear:
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_linear:
            nn.init.ones_(self.weight)

    def mean_center(self, input):
        return input - input.mean(-1, keepdim=True)

    def covariance(self, input):
        return 1 / self.normalized_shape[0] * input @ input.transpose(-1, -2)

    def symsqrtinv(self, matrix):
        _, s, v = matrix.svd()
        good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
        components = good.sum(-1)
        common = components.max()
        unbalanced = common != components.min()
        if common < s.size(-1):
            s = s[..., :common]
            v = v[..., :common]
            if unbalanced:
                good = good[..., :common]
        if unbalanced:
            s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
        return (v * 1 / torch.sqrt(s + self.eps).unsqueeze(-2)) @ v.transpose(-2, -1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.float64)
        input = self.mean_center(input)
        reg_matrix = torch.diag(torch.tensor([1.0, 2.0, 3.0])).unsqueeze(0).to(input.device).type(input.dtype)
        covar = self.covariance(input) + self.eps * reg_matrix
        covar_sqrtinv = self.symsqrtinv(covar)
        return (covar_sqrtinv @ input).to(self.weight.dtype) * self.weight.reshape(1, 1, self.normalized_shape[0])

    def extra_repr(self) -> str:
        return "{normalized_shape}, elementwise_linear={elementwise_linear}".format(**self.__dict__)