
from os.path import join
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_warn

from torchmdnet import datasets
from torchmdnet.utils import make_splits, MissingEnergyException, DataLoaderMasking
from torch_scatter import scatter
from torchmdnet.utils import collate_fn
import os



def init_noise_schemes(K=24):

    schemes = []

    for _ in range(K):
        schemes.append({
            "bond": np.random.uniform(0.005, 0.02),
            "angle": np.random.uniform(0.01, 0.03),
            "torsion": np.random.uniform(0.02, 0.06),
            "coord": np.random.uniform(0.001, 0.005)
        })
    return schemes


def update_noise_schemes(elite_schemes, K=24):

    # Gaussian (coord): ±0.001 | Bond angle: ±0.002 | Dihedral: ±0.004 | Bond: ±0.001
    new_local = []
    local_num = int(K * 0.4)
    for sch in elite_schemes[:local_num]:
        new_local.append({
            "bond": np.clip(sch["bond"] + np.random.uniform(-0.001, 0.001), 0.005, 0.02),
            "angle": np.clip(sch["angle"] + np.random.uniform(-0.002, 0.002), 0.01, 0.03),
            "torsion": np.clip(sch["torsion"] + np.random.uniform(-0.004, 0.004), 0.02, 0.06),
            "coord": np.clip(sch["coord"] + np.random.uniform(-0.001, 0.001), 0.001, 0.005)
        })


    new_global = init_noise_schemes(K=int(K * 0.1))


    new_schemes = elite_schemes + new_local + new_global
    return new_schemes[:K]


# ==============================================================================

# ==============================================================================
class DataModule(LightningDataModule):
    def __init__(self, hparams, dataset=None):
        super(DataModule, self).__init__()
        self._set_hparams(hparams.__dict__ if hasattr(hparams, "__dict__") else hparams)
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        self.dataset = dataset

        self.mask_atom = hparams.mask_atom
        self.model = hparams.model


        self.use_moes = getattr(self.hparams, 'use_moes', False)
        self.noise_pool_size = getattr(self.hparams, 'noise_pool_size', 24)  # 对齐你的K=24


        self.noise_schemes = init_noise_schemes(K=self.noise_pool_size)
        self.current_scheme_idx = 0

    def setup(self, stage):
        if self.dataset is None:
            # ==================================================================

            # ==================================================================
            if self.use_moes:
                def transform(data):

                    scheme = self.noise_schemes[self.current_scheme_idx % len(self.noise_schemes)]
                    self.current_scheme_idx += 1

                    data.pos_target = data.pos.clone()
                    data.pos = add_equi_noise_nnew(data, scheme)

                    if self.hparams["prior_model"] == "Atomref":
                        data.y = self.get_energy_data(data)
                    return data
            elif self.hparams['position_noise_scale'] > 0.:

                def transform(data):
                    noise = torch.randn_like(data.pos) * self.hparams['position_noise_scale']
                    data.pos_target = noise
                    data.pos = data.pos + noise
                    if self.hparams["prior_model"] == "Atomref":
                        data.y = self.get_energy_data(data)
                    return data
            else:
                transform = None

            # --------------------------

            # --------------------------
            if "LBADataset" in self.hparams["dataset"]:
                dataset_factory = getattr(datasets, self.hparams["dataset"])
                if self.hparams["dataset"] == 'LBADataset':
                    self.train_dataset = dataset_factory(
                        os.path.join(self.hparams["dataset_root"], "lba_train.npy"),
                        transform_noise=transform,
                        lp_sep=self.hparams['lp_sep'],
                        use_lig_feat=self.hparams['use_uni_feat'],
                        use_moes=self.use_moes
                    )
                    self.val_dataset = dataset_factory(
                        os.path.join(self.hparams["dataset_root"], "lba_valid.npy"),
                        transform_noise=None,
                        lp_sep=self.hparams['lp_sep'],
                        use_lig_feat=self.hparams['use_uni_feat'],
                        use_moes=self.use_moes
                    )
                    self.test_dataset = dataset_factory(
                        os.path.join(self.hparams["dataset_root"], "lba_test.npy"),
                        transform_noise=None,
                        lp_sep=self.hparams['lp_sep'],
                        use_lig_feat=self.hparams['use_uni_feat'],
                        use_moes=self.use_moes
                    )
                else:
                    self.train_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "train_data.pk"))
                    self.val_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "val_data.pk"))
                    self.test_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "test_data.pk"))

                if self.hparams["standardize"]:
                    self._standardize()
                return

            # --------------------------

            # --------------------------
            if self.hparams["dataset"] == "Custom":
                self.dataset = datasets.Custom(
                    self.hparams["coord_files"],
                    self.hparams["embed_files"],
                    self.hparams["energy_files"],
                    self.hparams["force_files"],
                    use_moes=self.use_moes
                )
            else:

                if 'BIAS' in self.hparams['dataset']:
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(
                        self.hparams["dataset_root"], self.hparams['sdf_path'],
                        self.hparams['position_noise_scale'], self.hparams['sample_number'],
                        self.hparams['violate'], dataset_arg=self.hparams["dataset_arg"],
                        transform=t, use_moes=self.use_moes
                    )
                elif 'PCQM4MV2_Force' in self.hparams['dataset']:
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(
                        self.hparams["dataset_root"], self.hparams['sdf_path'],
                        self.hparams['dihedral_angle_noise_scale'], self.hparams['angle_noise_scale'],
                        self.hparams['bond_length_scale'], dataset_arg=self.hparams["dataset_arg"],
                        transform=t, use_moes=self.use_moes
                    )
                elif 'Dihedral2' in self.hparams['dataset'] or 'Dihedral3' in self.hparams['dataset'] or 'Dihedral4' in \
                        self.hparams['dataset']:
                    add_radius_edge = True if self.hparams.model == 'painn' else False
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(
                        self.hparams["dataset_root"], self.hparams['sdf_path'],
                        self.hparams['dihedral_angle_noise_scale'], self.hparams['position_noise_scale'],
                        self.hparams['composition'], self.hparams['decay'], self.hparams['decay_coe'],
                        dataset_arg=self.hparams["dataset_arg"], equilibrium=self.hparams['equilibrium'],
                        eq_weight=self.hparams['eq_weight'], cod_denoise=self.hparams['cod_denoise'],
                        integrate_coord=self.hparams['integrate_coord'], addh=self.hparams['addh'],
                        mask_atom=self.hparams['mask_atom'], mask_ratio=self.hparams['mask_ratio'],
                        bat_noise=self.hparams['bat_noise'], transform=t, add_radius_edge=add_radius_edge,
                        use_moes=self.use_moes
                    )
                elif 'DihedralF' in self.hparams['dataset']:
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(
                        self.hparams["dataset_root"], self.hparams['sdf_path'],
                        self.hparams['dihedral_angle_noise_scale'], self.hparams['position_noise_scale'],
                        self.hparams["composition"], self.hparams['force_field'], self.hparams['pred_noise'],
                        cod_denoise=self.hparams['cod_denoise'], rdkit_conf=self.hparams['rdkit_conf'],
                        use_moes=self.use_moes
                    )
                elif 'Dihedral' in self.hparams['dataset']:
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(
                        self.hparams["dataset_root"], self.hparams['sdf_path'],
                        self.hparams['dihedral_angle_noise_scale'], self.hparams['position_noise_scale'],
                        self.hparams["composition"], dataset_arg=self.hparams["dataset_arg"],
                        transform=t, use_moes=self.use_moes
                    )
                elif 'QM9A' in self.hparams['dataset'] or 'MD17A' in self.hparams['dataset']:
                    transform_y = self.get_energy_data if (
                                self.hparams["prior_model"] == "Atomref" and 'QM9A' in self.hparams[
                            'dataset']) else None
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(
                        self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"],
                        transform=None, dihedral_angle_noise_scale=self.hparams['dihedral_angle_noise_scale'],
                        position_noise_scale=self.hparams['position_noise_scale'],
                        composition=self.hparams["composition"], transform_y=transform_y,
                        use_moes=self.use_moes
                    )
                elif 'TestData' in self.hparams['dataset']:
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(
                        self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"],
                        transform=None, use_moes=self.use_moes
                    )
                else:
                    add_radius_edge = True if self.hparams.model == 'painn' else False
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(
                        self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"],
                        transform=t, add_radius_edge=add_radius_edge,
                        use_moes=self.use_moes
                    )


                self.dataset_maybe_noisy = dataset_factory(transform)
                if self.hparams["prior_model"] == "Atomref":
                    def transform_atomref(data):
                        data.y = self.get_energy_data(data)
                        return data

                    self.dataset = dataset_factory(transform_atomref)
                else:
                    self.dataset = dataset_factory(None)


        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            self.hparams["train_size"],
            self.hparams["val_size"],
            self.hparams["test_size"],
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams["splits"],
        )
        print(f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}")

        self.train_dataset = Subset(self.dataset_maybe_noisy, self.idx_train)
        if self.hparams['denoising_only']:
            self.val_dataset = Subset(self.dataset_maybe_noisy, self.idx_val)
            self.test_dataset = Subset(self.dataset_maybe_noisy, self.idx_test)
        else:
            self.val_dataset = Subset(self.dataset, self.idx_val)
            self.test_dataset = Subset(self.dataset, self.idx_test)

        if hasattr(self.hparams, "infer_mode"):
            self.test_dataset = self.dataset

        if self.hparams["standardize"]:
            self._standardize()

    # ==================================================================

    # ==================================================================
    def update_noise_schemes(self, elite_schemes):
        self.noise_schemes = update_noise_schemes(elite_schemes, K=self.noise_pool_size)
        self.current_scheme_idx = 0

        self._saved_dataloaders.clear()

    # ==================================================================

    # ==================================================================
    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        loaders = [self._get_dataloader(self.val_dataset, "val")]
        if (
                len(self.test_dataset) > 0
                and self.trainer.current_epoch % self.hparams["test_interval"] == 0 and self.trainer.current_epoch != 0
        ):
            loaders.append(self._get_dataloader(self.test_dataset, "test"))
        return loaders

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def atomref(self):
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        store_dataloader = (
                store_dataloader and not self.trainer.reload_dataloaders_every_n_epochs
        )
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False

        if self.mask_atom:
            dl = DataLoaderMasking(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.hparams["num_workers"],
                pin_memory=True,
            )
        else:
            if self.model == 'egnn':
                from torch.utils.data import DataLoader
                dl = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=self.hparams["num_workers"],
                    pin_memory=True,
                    collate_fn=collate_fn,
                )
            else:
                from torch_geometric.data import DataLoader
                dl = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=self.hparams["num_workers"],
                    pin_memory=True,
                )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl

    def get_energy_data(self, data):
        if data.y is None:
            raise MissingEnergyException()
        atomref_energy = self.atomref.squeeze()[data.z].sum()
        return (data.y.squeeze() - atomref_energy).unsqueeze(dim=0).unsqueeze(dim=1)

    def _standardize(self):
        def get_energy(batch, atomref):
            if batch.y is None:
                raise MissingEnergyException()
            if atomref is None:
                return batch.y.clone()
            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.squeeze() - atomref_energy.squeeze()).clone()

        data = tqdm(
            self._get_dataloader(self.train_dataset, "val", store_dataloader=False),
            desc="computing mean and std",
        )
        try:
            atomref = None
            ys = torch.cat([get_energy(batch, atomref) for batch in data])
        except MissingEnergyException:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset only contains forces."
            )
            return

        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)