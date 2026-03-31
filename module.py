import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LRScheduler
from abc import ABCMeta, abstractmethod
from scipy.stats import spearmanr, pearsonr
from pytorch_lightning import LightningModule
from torch.nn.functional import mse_loss, l1_loss, smooth_l1_loss

from utils import load_model, create_model

__all__ = ["LNNP", "CustomScheduler"]


class LNNP(LightningModule):
    def __init__(self, hparams, prior_model=None, mean=None, std=None):
        super(LNNP, self).__init__()
        self.save_hyperparameters(hparams)

        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, args=self.hparams)
        elif self.hparams.pretrained_model:
            if hasattr(self.hparams, 'infer_mode') and self.hparams.infer_mode:
                mean = None
                std = None
            self.model = load_model(self.hparams.pretrained_model, args=self.hparams, mean=mean, std=std)
        else:
            self.model = create_model(self.hparams, prior_model, mean, std)

        self.ema = None
        self._reset_ema_dict()

        self.losses = None
        self._reset_losses_dict()

        self.sep_noisy_node = self.hparams.sep_noisy_node
        self.train_loss_type = self.hparams.train_loss_type

        if self.hparams.mask_atom:
            self.criterion = torch.nn.CrossEntropyLoss()

        self.bond_length_scale = self.hparams.bond_length_scale
        self.dataset = self.hparams['dataset']

        # =======================================
        self.use_moes = getattr(self.hparams, 'use_moes', False)

        self.motif_weight = getattr(self.hparams, 'motif_weight', 0.1)
        self.influence_weight = getattr(self.hparams, 'influence_weight', 0.05)
        self.ens_weight = getattr(self.hparams, 'ens_weight', 0.1)

        self.noise_update_interval = getattr(self.hparams, 'noise_update_interval', 2)

        self.epoch_scheme_scores = []

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.lr_schedule == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, self.hparams.lr_cosine_length)
            lr_scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        elif self.hparams.lr_schedule == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(optimizer, "min", factor=self.hparams.lr_factor,
                                          patience=self.hparams.lr_patience, min_lr=self.hparams.lr_min)
            lr_scheduler = {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch", "frequency": 1}
        elif self.hparams.lr_schedule == "cosine_warmup":
            scheduler = CustomScheduler(optimizer=optimizer, max_lr=self.hparams.lr, min_lr=self.hparams.lr_min,
                                        iters_per_epoch=len(self.train_dataloader()),
                                        num_epochs=self.hparams.num_epochs)
            lr_scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        else:
            raise ValueError(f"Unknown lr_schedule: {self.hparams.lr_schedule}")
        return [optimizer], [lr_scheduler]

    # ====================== ======================
    def forward(self, z, pos, batch=None, batch_org=None, egnn_dict=None, radius_edge_index=None,
                motif_feat=None, motif_mask=None, influence_matrix=None, pos_gt=None):
        if self.hparams.model == 'egnn':
            return self.model(z, pos, egnn_dict=egnn_dict)
        elif self.hparams.model == 'painn':
            return self.model(z, pos, batch=batch, batch_org=batch_org, radius_edge_index=radius_edge_index)
        else:

            return self.model(z, pos, batch=batch, batch_org=batch_org,
                              motif_feat=motif_feat, motif_mask=motif_mask,
                              influence_matrix=influence_matrix, pos_gt=pos_gt)

    def training_step(self, batch, batch_idx):
        if self.train_loss_type == 'smooth_l1_loss':
            return self.step(batch, smooth_l1_loss, 'train')
        elif self.train_loss_type == 'l1_loss':
            return self.step(batch, l1_loss, "train")
        return self.step(batch, mse_loss, "train")

    def validation_step(self, batch, batch_idx, *args):
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            return self.step(batch, l1_loss, "val")
        return self.step(batch, l1_loss, "test")

    def test_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, "test")

    def process_batch_idx(self, batch):
        batch_info = batch['batch']
        batch_num = batch._num_graphs
        slice_dict = batch._slice_dict
        bond_target_indx = slice_dict['bond_target']
        angle_target_indx = slice_dict['angle_target']
        dihedral_target_indx = slice_dict['dihedral_target']
        rotate_dihedral_target_indx = slice_dict['rotate_dihedral_target']
        for i in range(batch_num):
            cur_num = slice_dict['pos'][i]
            batch.bond_target[bond_target_indx[i]:bond_target_indx[i + 1]][:, :2] += cur_num
            batch.angle_target[angle_target_indx[i]:angle_target_indx[i + 1]][:, :3] += cur_num
            batch.dihedral_target[dihedral_target_indx[i]:dihedral_target_indx[i + 1]][:, :4] += cur_num
            batch.rotate_dihedral_target[rotate_dihedral_target_indx[i]:rotate_dihedral_target_indx[i + 1]][:,
            :4] += cur_num

    def step(self, batch, loss_fn, stage):
        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            if stage == 'test' and 'org_pos' in batch.keys():
                # ====================== 【仅调整】模型输出解包（6个值，严格对齐） ======================
                pred, noise_pred, deriv, mask_logits, sk_score, ens_loss = self(
                    batch.z, batch.org_pos, batch.batch,
                    motif_feat=batch.get("motif_feat", None),
                    motif_mask=batch.get("motif_mask", None),
                    influence_matrix=batch.get("influence_matrix", None),
                    pos_gt=batch.get("pos_target", None)
                )
            else:
                if self.sep_noisy_node:

                    pred, _, deriv, _, _, _ = self(batch.z, batch.org_pos, batch.batch)
                    _, noise_pred, _, _, _, _ = self(batch.z, batch.pos, batch.batch)
                else:
                    if self.bond_length_scale > 0:
                        self.process_batch_idx(batch)

                    # ====================== 【仅调整】模型输出解包（6个值，严格对齐） ======================
                    pred, noise_pred, deriv, mask_logits, sk_score, ens_loss = self(
                        batch.z, batch.pos, batch.batch, batch_org=batch,
                        motif_feat=batch.get("motif_feat", None),
                        motif_mask=batch.get("motif_mask", None),
                        influence_matrix=batch.get("influence_matrix", None),
                        pos_gt=batch.get("pos_target", None)
                    )

        denoising_is_on = ("pos_target" in batch or "bond_target" in batch) and (
                    self.hparams.denoising_weight > 0) and (noise_pred is not None)

        loss_y, loss_dy, loss_pos, mask_atom_loss = 0, 0, 0, 0
        loss_motif, loss_influence = 0.0, 0.0


        if self.hparams.mask_atom:
            mask_indices = batch['masked_atom_indices']
            mask_logits = mask_logits[mask_indices]
            mask_atom_loss = self.criterion(mask_logits, batch.mask_node_label)
            self.losses[stage + "_mask_atom_loss"].append(mask_atom_loss.detach())


        if self.hparams.derivative:
            if "y" not in batch:
                deriv = deriv + pred.sum() * 0
            if stage == "test" and "smi" in batch:
                loss_dy = loss_fn(deriv, batch.dy)
                y_mae = l1_loss(pred, batch.y)
                y_rmse = torch.sqrt(mse_loss(pred, batch.y))
                self.losses["test_y_mae"].append(y_mae.detach())
                self.losses["test_y_rmse"].append(y_rmse.detach())
            else:
                loss_dy = loss_fn(deriv, batch.dy)
            if torch.isnan(loss_dy).sum():
                print('loss nan happens')

            if stage in ["train", "val"] and self.hparams.ema_alpha_dy < 1:
                if self.ema[stage + "_dy"] is None:
                    self.ema[stage + "_dy"] = loss_dy.detach()
                loss_dy = self.hparams.ema_alpha_dy * loss_dy + (1 - self.hparams.ema_alpha_dy) * self.ema[
                    stage + "_dy"]
                self.ema[stage + "_dy"] = loss_dy.detach()
            if self.hparams.force_weight > 0:
                self.losses[stage + "_dy"].append(loss_dy.detach())


        if "y" in batch:
            if (noise_pred is not None) and not denoising_is_on:
                pred = pred + noise_pred.sum() * 0
            if batch.y.ndim == 1:
                batch.y = batch.y.unsqueeze(1)
            if torch.isnan(pred).sum():
                print('pred nan happends')
            if stage == "test" and "smi" in batch:
                loss_y = loss_fn(pred, batch.y)
                dy_mae = l1_loss(deriv, batch.dy)
                dy_rmse = torch.sqrt(mse_loss(deriv, batch.dy))
                self.losses["test_dy_mae"].append(dy_mae.detach())
                self.losses["test_dy_rmse"].append(dy_rmse.detach())
            else:
                if hasattr(self.hparams, 'infer_mode'):
                    new_res = pred.squeeze().tolist()
                    if isinstance(new_res, list):
                        self.losses['pred_values'].extend(new_res)
                    else:
                        self.losses['pred_values'].append(new_res)
                loss_y = loss_fn(pred, batch.y)

            if 'LBADataset' in self.dataset and stage in ['val', 'test']:
                if stage == 'test':
                    self.losses['lba_pred_org'].append(pred.detach())
                    self.losses['lba_y_org'].append(batch.y)
                    batch_size = batch.y.size(0)
                    noise_pock_mean = []
                    noise_lig_mean = []
                    noise_all_mean = []
                    for i in range(batch_size):
                        noise_pred_ele = noise_pred[batch.batch == i]
                        noise_pred_ele_pocket = noise_pred_ele[:batch.pocket_atomsnum[i]]
                        noise_pred_ele_lig = noise_pred_ele[batch.pocket_atomsnum[i]:]
                        noise_all_mean.append(noise_pred_ele.mean())
                        noise_pock_mean.append(noise_pred_ele_pocket.mean())
                        noise_lig_mean.append(noise_pred_ele_lig.mean())
                    self.losses['noise_all_mean'].append(torch.tensor(noise_all_mean))
                    self.losses['noise_pock_mean'].append(torch.tensor(noise_pock_mean))
                    self.losses['noise_lig_mean'].append(torch.tensor(noise_lig_mean))
                else:
                    self.losses['lba_pred_org_val'].append(pred.detach())
                    self.losses['lba_y_org_val'].append(batch.y)

            if torch.isnan(loss_y).sum():
                print('loss nan happens')
            if stage in ["train", "val"] and self.hparams.ema_alpha_y < 1:
                if self.ema[stage + "_y"] is None:
                    self.ema[stage + "_y"] = loss_y.detach()
                loss_y = self.hparams.ema_alpha_y * loss_y + (1 - self.hparams.ema_alpha_y) * self.ema[stage + "_y"]
                self.ema[stage + "_y"] = loss_y.detach()
            if self.hparams.energy_weight > 0:
                self.losses[stage + "_y"].append(loss_y.detach())


        if denoising_is_on:
            def weighted_mse_loss(input, target, weight):
                return (weight.reshape(-1, 1).repeat((1, 3)) * (input - target) ** 2).mean()

            def custom_mse_loss(input, target):
                return ((input - target) ** 2).mean()

            if 'wg' in batch.keys():
                loss_fn = weighted_mse_loss
                wt = batch['w1'].sum() / batch['idx'].shape[0]
                weights = batch['wg'] / wt
            else:
                loss_fn = custom_mse_loss
            if self.model.pos_normalizer is not None:
                if self.hparams.model == 'egnn':
                    normalized_pos_target = batch.pos_target.reshape(-1, 3)[atom_mask.squeeze()]
                    normalized_pos_target = self.model.pos_normalizer(normalized_pos_target)
                    noise_pred = noise_pred[atom_mask.squeeze()]
                    loss_pos = loss_fn(noise_pred, normalized_pos_target)
                else:
                    normalized_pos_target = self.model.pos_normalizer(batch.pos_target)
                    loss_pos = loss_fn(noise_pred, normalized_pos_target, weights) if 'wg' in batch.keys() else loss_fn(
                        noise_pred, normalized_pos_target)
                self.losses[stage + "_pos"].append(loss_pos.detach())
            elif self.model.bond_pos_normalizer is not None:
                normalized_bond_target = self.model.bond_pos_normalizer(batch.bond_target[:, -1])
                normalized_angle_target = self.model.angle_pos_normalizer(batch.angle_target[:, -1])
                normalized_dihedral_target = self.model.dihedral_pos_normalizer(batch.dihedral_target[:, -1])
                normalized_rotate_dihedral_target = self.model.rotate_dihedral_pos_normalizer(
                    batch.rotate_dihedral_target[:, -1])
                loss_bond = loss_fn(noise_pred[0], normalized_bond_target)
                loss_angle = loss_fn(noise_pred[1], normalized_angle_target)
                loss_dihedral = loss_fn(noise_pred[2], normalized_dihedral_target)
                loss_rotate_dihedral = loss_fn(noise_pred[3], normalized_rotate_dihedral_target)
                self.losses[stage + "_bond"].append(loss_bond.detach())
                self.losses[stage + "_angle"].append(loss_angle.detach())
                self.losses[stage + "_dihedral"].append(loss_dihedral.detach())
                self.losses[stage + "_rotate_dihedral"].append(loss_rotate_dihedral.detach())
                loss_pos = loss_bond + loss_angle + loss_dihedral + loss_rotate_dihedral
            else:
                loss_pos = loss_fn(noise_pred, batch.pos_target, weights) if 'wg' in batch.keys() else loss_fn(
                    noise_pred, batch.pos_target)
                self.losses[stage + "_pos"].append(loss_pos.detach())

        # =================== ======================
        if self.use_moes:

            if "motif_y" in batch and motif_pred is not None:
                loss_motif = loss_fn(motif_pred[batch.motif_mask], batch.motif_y)
                self.losses[stage + "_motif"].append(loss_motif.detach())

            if batch.get("influence_matrix", None) is not None:
                loss_influence = torch.norm(batch.influence_matrix, p=2)
                self.losses[stage + "_influence"].append(loss_influence.detach())

            if ens_loss is not None:
                self.losses[stage + "_ens"].append(ens_loss.detach())

            if stage == "train" and sk_score is not None:
                self.epoch_scheme_scores.append(sk_score.mean().detach().cpu().item())


        loss = (
                loss_y * self.hparams.energy_weight
                + loss_dy * self.hparams.force_weight
                + loss_pos * self.hparams.denoising_weight
                + mask_atom_loss
                + loss_motif * self.motif_weight
                + loss_influence * self.influence_weight
                + (ens_loss if ens_loss is not None else 0.0) * self.ens_weight
        )

        self.losses[stage].append(loss.detach())

        if stage == 'train':
            train_metrics = {k + "_per_step": v[-1] for k, v in self.losses.items() if
                             (k.startswith("train") and len(v) > 0)}
            train_metrics['lr_per_step'] = self.trainer.optimizers[0].param_groups[0]["lr"]
            train_metrics['step'] = self.trainer.global_step
            self.log_dict(train_metrics, sync_dist=True)

        return loss

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.hparams.lr_warmup_steps))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    # ======================================
    def on_train_epoch_end(self):
        if not self.use_moes:
            return
        #
        if self.current_epoch % self.noise_update_interval != 0:
            return
        #
        if len(self.epoch_scheme_scores) == 0:
            return


        scheme_scores = torch.tensor(self.epoch_scheme_scores, device=self.device)
        sorted_indices = torch.argsort(scheme_scores)


        datamodule = self.trainer.datamodule
        total_schemes = len(datamodule.noise_schemes)
        elite_count = total_schemes // 2
        elite_indices = sorted_indices[:elite_count]
        elite_schemes = [datamodule.noise_schemes[i] for i in elite_indices]


        datamodule.update_noise_schemes(elite_schemes)

        # 日志输出
        self.log("moes/noise_schemes_updated", 1.0, prog_bar=True, sync_dist=True)
        self.log("moes/elite_scheme_count", float(len(elite_schemes)), prog_bar=True, sync_dist=True)

        # 清空分数缓存
        self.epoch_scheme_scores.clear()

    def training_epoch_end(self, training_step_outputs):
        dm = self.trainer.datamodule
        if hasattr(dm, "test_dataset") and len(dm.test_dataset) > 0:
            should_reset = (self.current_epoch % self.hparams.test_interval == 0 or (
                        self.current_epoch - 1) % self.hparams.test_interval == 0)
            if should_reset:
                self.trainer.reset_val_dataloader(self)

    def compute_metrics_lba(self, predictions, targets):
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        spearman_corr, _ = spearmanr(predictions, targets)
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        pearson_corr, _ = pearsonr(predictions.flatten(), targets.flatten())
        return spearman_corr, rmse, pearson_corr

    def test_epoch_end(self, outputs):
        if hasattr(self.hparams, 'infer_mode') and self.hparams.infer_mode:
            with open(self.hparams.output_file, 'w') as f:
                for item in self.losses['pred_values']:
                    f.write("%s\n" % item)
        result_dict = {}
        if len(self.losses["test_y"]) > 0:
            result_dict["test_loss_y"] = torch.stack(self.losses["test_y"]).mean()
        return result_dict

    def validation_epoch_end(self, validation_step_outputs):
        if not self.trainer.sanity_checking:
            result_dict = {
                "epoch": self.current_epoch,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": torch.stack(self.losses["train"]).mean(),
                "val_loss": torch.stack(self.losses["val"]).mean(),
            }

            if len(self.losses["test"]) > 0:
                result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()
            if len(self.losses["train_y"]) > 0 and len(self.losses["train_dy"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
                result_dict["train_loss_dy"] = torch.stack(self.losses["train_dy"]).mean()
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
                result_dict["val_loss_dy"] = torch.stack(self.losses["val_dy"]).mean()
            if len(self.losses["train_y"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
            if len(self.losses['val_y']) > 0:
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
            if len(self.losses["test_y"]) > 0:
                result_dict["test_loss_y"] = torch.stack(self.losses["test_y"]).mean()

            # MOES 损失日志
            if self.use_moes:
                if len(self.losses["train_motif"]) > 0:
                    result_dict["train_motif_loss"] = torch.stack(self.losses["train_motif"]).mean()
                    result_dict["val_motif_loss"] = torch.stack(self.losses["val_motif"]).mean()
                if len(self.losses["train_influence"]) > 0:
                    result_dict["train_influence_loss"] = torch.stack(self.losses["train_influence"]).mean()
                if len(self.losses["train_ens"]) > 0:
                    result_dict["train_ens_loss"] = torch.stack(self.losses["train_ens"]).mean()
                    result_dict["val_ens_loss"] = torch.stack(self.losses["val_ens"]).mean()

            self.log_dict(result_dict, sync_dist=True)
        self._reset_losses_dict()

    def _reset_losses_dict(self):
        self.losses = {
            "train": [], "val": [], "test": [],
            "train_y": [], "val_y": [], "test_y": [],
            "train_dy": [], "val_dy": [], "test_dy": [],
            "train_pos": [], "val_pos": [], "test_pos": [],
            "train_mask_atom_loss": [], "val_mask_atom_loss": [], "test_mask_atom_loss": [],
            "train_bond": [], "val_bond": [], "test_bond": [],
            "train_angle": [], "val_angle": [], "test_angle": [],
            "train_dihedral": [], "val_dihedral": [], "test_dihedral": [],
            "train_rotate_dihedral": [], "val_rotate_dihedral": [], "test_rotate_dihedral": [],
            "test_y_mae": [], "test_y_rmse": [], "test_dy_mae": [], "test_dy_rmse": [],
            "lba_pred_org": [], "lba_pred_org_val": [], "lba_y_org": [], "lba_y_org_val": [],
            "noise_all_mean": [], "noise_pock_mean": [], "noise_lig_mean": [],
            "pred_values": [],

            # MOES-Pred 专属损失
            "train_motif": [], "val_motif": [], "test_motif": [],
            "train_influence": [], "val_influence": [], "test_influence": [],
            "train_ens": [], "val_ens": [], "test_ens": [],
        }

    def _reset_ema_dict(self):
        self.ema = {"train_y": None, "val_y": None, "train_dy": None, "val_dy": None}


class CustomScheduler(LRScheduler):
    def __init__(self, optimizer, max_lr, min_lr, iters_per_epoch, num_epochs, last_epoch=-1):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.iters_per_epoch = iters_per_epoch
        self.num_epochs = num_epochs
        self.total_iters = iters_per_epoch * num_epochs
        self.warmup_epoch = 0.3
        self.patience_epoch = 0.7
        super(CustomScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cur_iter = self.last_epoch
        if cur_iter < self.warmup_epoch * self.iters_per_epoch:
            return [self.max_lr * cur_iter / (self.warmup_epoch * self.total_iters) for _ in self.base_lrs]
        elif cur_iter < (self.patience_epoch + self.warmup_epoch) * self.iters_per_epoch:
            return [self.max_lr for _ in self.base_lrs]
        else:
            prev_iters = (self.patience_epoch + self.warmup_epoch) * self.iters_per_epoch
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * \
                 (1. + math.cos(math.pi * (cur_iter - prev_iters) / (self.total_iters - prev_iters)))
            return [lr for _ in self.base_lrs]