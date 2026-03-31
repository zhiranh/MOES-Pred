from typing import Optional, Callable, List
from pathlib import Path
import sys
import urllib.request
import zipfile
import os
from tqdm import tqdm
import glob
#import ase
import numpy as np
from rdkit import Chem
from torchmdnet.utils import isRingAromatic, get_geometry_graph_ring
from typing import Any, Callable, List, Optional, Tuple, Union
from collections.abc import Sequence
from torch import Tensor
IndexType = Union[slice, Tensor, np.ndarray, Sequence]
import random
import torch.nn.functional as F
import copy
import lmdb
import pickle

from typing import List, Optional, Callable
import zipfile


import numpy as np


from rdkit.Chem import GetPeriodicTable

import urllib.request
import shutil
import torch
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)

from torsion_utils import get_torsions, GetDihedral, apply_changes, get_rotate_order_info, add_equi_noise, add_equi_noise_new
from rdkit.Geometry import Point3D
from torch_geometric.nn import radius_graph

def download_url(url: str, folder: str, fname: str = None):
    """极简版 download_url，兼容 PyG 接口。"""
    os.makedirs(folder, exist_ok=True)
    if fname is None:
        fname = os.path.basename(url)
    fpath = os.path.join(folder, fname)
    if os.path.exists(fpath):          # 已存在就跳过
        return fpath
    # 下载并显示进度
    with urllib.request.urlopen(url) as resp, open(fpath, 'wb') as out_file:
        shutil.copyfileobj(resp, out_file)
    return fpath

class PCQM4MV2_XYZ(InMemoryDataset):
    r"""3D coordinates for molecules in the PCQM4Mv2 dataset (from zip)."""

    raw_url = 'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2_xyz.zip'  # 1. 去空格

    def __init__(self, root: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 dataset_arg: Optional[str] = None):
        assert dataset_arg is None, "PCQM4MV2 does not take any dataset args."
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['pcqm4m-v2_xyz.zip']

    @property
    def processed_file_names(self) -> str:
        return 'pcqm4mv2_xyz.pt'

    # ---------- 2. 健壮解压 ----------
    def download(self):
        pass

    # ---------- 3. 处理 ----------
    def process(self):
        xyz_dir = os.path.join(self.raw_dir, 'pcqm4m-v2_xyz')
        files = sorted(glob.glob(os.path.join(xyz_dir, '**', '*.xyz'), recursive=True))
        if not files:
            raise FileNotFoundError(f'no *.xyz found under {xyz_dir} (recursive)')

        data_list = []
        for idx, fn in enumerate(tqdm(files, ncols=100)):
            with open(fn) as f:
                lines = f.readlines()
            natom = int(lines[0])
            atom_type, coords = [], []
            for line in lines[2: 2 + natom]:
                sym, x, y, z = line.split()[:4]
                atom_type.append(GetPeriodicTable().GetAtomicNumber(sym))
                coords.append([float(x), float(y), float(z)])

            z = torch.tensor(atom_type, dtype=torch.long)
            pos = torch.tensor(coords, dtype=torch.float)
            data = Data(z=z, pos=pos, idx=idx)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


class PCQM4MV2_XYZ2(InMemoryDataset):
    r"""3D coordinates for molecules in the PCQM4Mv2 dataset (from zip)."""

    raw_url = 'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2_xyz.zip'  # 1. 去除空格

    def __init__(self, root: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 dataset_arg: Optional[str] = None,
                 use_lmdb_fallback: bool = True):  # 2. 新增参数：是否启用LMDB回退模式
        assert dataset_arg is None, "PCQM4MV2 does not take any dataset args."

        # 3. 核心改进：在调用父类之前检查是否可以跳过处理
        self.root = Path(root).absolute()
        self.use_lmdb_fallback = use_lmdb_fallback

        # 检查是否存在已处理的文件或LMDB
        self._can_skip_process = self._check_skip_process()

        # 如果满足跳过条件，手动初始化必要属性，避免触发父类 process()
        if self._can_skip_process:
            self._manual_init()
        else:
            # 否则走标准流程
            super().__init__(str(self.root), transform, pre_transform, pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[0])

    def _check_skip_process(self) -> bool:
        """检查是否可以跳过 process() 流程"""
        # 情况1: 已存在处理好的 .pt 文件
        if (self.root / 'processed' / self.processed_file_names).exists():
            print(f"Found processed file, loading directly.")
            return True

        # 情况2: 启用LMDB回退且LMDB存在
        lmdb_path = self.root / 'MOL_LMDB'
        if self.use_lmdb_fallback and lmdb_path.exists():
            print(f"XYZ files not found, but LMDB exists at {lmdb_path}. Using LMDB mode.")
            return True

        return False

    def _manual_init(self):
        """手动初始化，绕过 InMemoryDataset 的 process() 调用"""
        # 加载已处理的数据（如果存在）
        if (self.root / 'processed' / self.processed_file_names).exists():
            self.data, self.slices = torch.load(self.root / 'processed' / self.processed_file_names)
        else:
            # 否则创建空数据，实际数据将通过 __getitem__ 从LMDB动态加载
            self.data = Data()
            self.slices = {}

        self.transform = None
        self.pre_transform = None
        self.pre_filter = None

    @property
    def raw_file_names(self) -> List[str]:
        return ['pcqm4m-v2_xyz.zip']

    @property
    def processed_file_names(self) -> str:
        return 'pcqm4mv2_xyz.pt'

    # ---------- 2. 实现自动下载 ----------
    def download(self):
        """自动下载并解压 xyz 文件"""
        if self._can_skip_process:
            print("Skipping download due to existing processed data or LMDB.")
            return

        raw_dir = self.root / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)

        zip_path = raw_dir / 'pcqm4m-v2_xyz.zip'

        # 如果已下载，跳过
        if zip_path.exists():
            print(f"Found existing zip file: {zip_path}")
        else:
            print(f"Downloading PCQM4MV2 XYZ data to {zip_path}...")
            try:
                # 显示下载进度
                def progress_hook(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = min(downloaded * 100 / total_size, 100)
                    sys.stdout.write(f'\rDownloading: {percent:.1f}%')
                    sys.stdout.flush()

                urllib.request.urlretrieve(self.raw_url, zip_path, reporthook=progress_hook)
                print("\nDownload complete.")
            except Exception as e:
                raise RuntimeError(f"Failed to download from {self.raw_url}: {e}")

        # 解压
        extract_dir = raw_dir / 'pcqm4m-v2_xyz'
        if extract_dir.exists() and list(extract_dir.glob('*.xyz')):
            print(f"XYZ files already extracted to {extract_dir}")
        else:
            print(f"Extracting to {extract_dir}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # 显示解压进度
                    members = zip_ref.infolist()
                    for member in tqdm(members, desc="Extracting"):
                        zip_ref.extract(member, raw_dir)
                print("Extraction complete.")
            except Exception as e:
                raise RuntimeError(f"Failed to extract {zip_path}: {e}")

    # ---------- 3. 健壮的 process ----------
    def process(self):
        """处理 XYZ 文件为 PyG 格式"""
        if self._can_skip_process:
            print("Skipping process due to existing data or LMDB mode.")
            return

        xyz_dir = self.root / 'raw' / 'pcqm4m-v2_xyz'
        files = sorted(xyz_dir.glob('**/*.xyz'))

        if not files:
            # 4. 增强错误提示，指导用户
            error_msg = f"""
            No *.xyz files found under {xyz_dir}.

            Possible solutions:
            1. Ensure the zip file is downloaded and extracted to: {xyz_dir}
            2. If you have LMDB data, set use_lmdb_fallback=True in dataset init
            3. Run download() first: 
               dataset = PCQM4MV2_XYZ(root='your_path')
               dataset.download()

            Expected structure:
            {xyz_dir}/
                ├── dsgdb9nsd_000001.xyz
                ├── dsgdb9nsd_000002.xyz
                └── ...
            """
            raise FileNotFoundError(error_msg.strip())

        print(f"Found {len(files)} XYZ files. Processing...")
        data_list = []
        for idx, fn in enumerate(tqdm(files, desc="Processing XYZ", ncols=100)):
            try:
                with open(fn) as f:
                    lines = f.readlines()
                if len(lines) < 3:
                    print(f"Warning: {fn} is malformed, skipping.")
                    continue

                natom = int(lines[0].strip())
                atom_type, coords = [], []
                for line in lines[2: 2 + natom]:
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    sym, x, y, z = parts[:4]
                    try:
                        atom_type.append(GetPeriodicTable().GetAtomicNumber(sym))
                        coords.append([float(x), float(y), float(z)])
                    except (ValueError, KeyError):
                        print(f"Warning: Invalid data in {fn}, skipping line: {line}")
                        continue

                if len(atom_type) == 0:
                    print(f"Warning: No valid atoms in {fn}, skipping.")
                    continue

                z = torch.tensor(atom_type, dtype=torch.long)
                pos = torch.tensor(coords, dtype=torch.float)
                data = Data(z=z, pos=pos, idx=idx)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
            except Exception as e:
                print(f"Error processing {fn}: {e}, skipping.")
                continue

        if not data_list:
            raise RuntimeError("No valid molecules were processed!")

        # 保存处理好的数据
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.collate(data_list), self.processed_paths[0])
        print(f"Saved processed data to {self.processed_paths[0]}")




# Globle variable
MOL_LST = None
MOL_DEBUG_LST = None
debug = False
debug_cnt = 0





# use force filed definition
# bond length, angle ,dihedral angel
# equilibrium
EQ_MOL_LST = None
EQ_EN_LST = None

class PCQM4MV2_Dihedral2(PCQM4MV2_XYZ):
    '''
    We process the data1 by adding noise to atomic coordinates and providing denoising targets for denoising pre-training.
    '''
    def __init__(self, root: str, sdf_path: str, dihedral_angle_noise_scale: float, position_noise_scale: float, composition: bool, decay=False, decay_coe=0.2, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, dataset_arg: Optional[str] = None, equilibrium=False, eq_weight=False, cod_denoise=False, integrate_coord=False, addh=False, mask_atom=False, mask_ratio=0.15, bat_noise=False, add_radius_edge=False):
        assert dataset_arg is None, "PCQM4MV2_Dihedral does not take any dataset args."
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.suppl = Chem.SDMolSupplier(sdf_path)
        self.dihedral_angle_noise_scale = dihedral_angle_noise_scale
        self.position_noise_scale = position_noise_scale
        self.composition = composition # angle noise as the start

        self.decay = decay
        self.decay_coe = decay_coe

        self.random_pos_prb = 0.5
        self.equilibrium = equilibrium # equilibrium settings
        self.eq_weight = eq_weight
        self.cod_denoise = cod_denoise # reverse to coordinate denoise

        self.integrate_coord = integrate_coord
        self.addh = addh

        self.mask_atom = mask_atom
        self.mask_ratio = mask_ratio
        self.num_atom_type = 119

        self.bat_noise = bat_noise

        global MOL_LST
        global EQ_MOL_LST
        global EQ_EN_LST

        if self.equilibrium and EQ_MOL_LST is None:
            # debug
            EQ_MOL_LST = np.load('MG_MOL_All.npy', allow_pickle=True) # mol lst
            EQ_EN_LST = np.load('MG_All.npy', allow_pickle=True) # energy lst
        else:
            #选用这个
            if MOL_LST is None:
            # import pickle
            # with open(sdf_path, 'rb') as handle:
            #     MOL_LST = pickle.load(handle)
            # MOL_LST = np.load("mol_iter_all.npy", allow_pickle=True)
                # MOL_LST = np.load("h_mol_lst.npy", allow_pickle=True)
                MOL_LST = lmdb.open(f'{root}/MOL_LMDB', readonly=True, subdir=True, lock=False)

        if debug:
            global MOL_DEBUG_LST
            if MOL_DEBUG_LST is None:
                # MOL_DEBUG_LST = Chem.SDMolSupplier("pcqm4m-v2-train.sdf")
                MOL_DEBUG_LST = np.load("mol_iter_all.npy", allow_pickle=True)

        self.add_radius_edge = add_radius_edge
        if self.add_radius_edge:
            self.radius = 5.0

    def transform_noise(self, data, position_noise_scale):
        noise = torch.randn_like(torch.tensor(data)) * position_noise_scale
        data_noise = data + noise.numpy()
        return data_noise

    def transform_noise_decay(self, data, position_noise_scale, decay_coe_lst):
        noise = torch.randn_like(torch.tensor(data)) * position_noise_scale * torch.tensor(decay_coe_lst)
        data_noise = data + noise.numpy()
        return data_noise

    def __getitem__(self, idx: Union[int, np.integer, IndexType]) -> Union['Dataset', Data]:
        """
        Retrieves and processes a data1 item at the specified index, adding noise to atomic coordinates and providing
        denoising targets for denoising pre-training.

        Args:
            idx (Union[int, np.integer, IndexType]): Index of the data1 item to retrieve.

        Returns:
            Union['Dataset', Data]: Processed data1 item with original and noisy coordinates, and denoising targets.

        Notes:
            When processing data1, if `bat_noise` is enabled, the 'bond angle torsion noise' is added to the molecule's equilibrium
            conformation. Otherwise, dihedral angle noise is applied. Gaussian coordinate noise is subsequently added.
        """
        org_data = super().__getitem__(idx)
        org_atom_num = org_data.pos.shape[0]
        # change org_data coordinate
        # get mol

        # check whether mask or not
        if self.mask_atom:
            num_atoms = org_data.z.size(0)
            sample_size = int(num_atoms * self.mask_ratio + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)
            org_data.mask_node_label = org_data.z[masked_atom_indices]
            org_data.z[masked_atom_indices] = self.num_atom_type
            org_data.masked_atom_indices = torch.tensor(masked_atom_indices)

        #“equilibrium 模式：随机挑一个低能构象并给权重；普通模式：LMDB 里直接反序列化分子。

        if self.equilibrium:
            # for debug
            # max_len = 422325 - 1
            # idx = idx.item() % max_len
            idx = idx.item()
            mol = copy.copy(EQ_MOL_LST[idx])
            energy_lst = EQ_EN_LST[idx]
            eq_confs = len(energy_lst)
            conf_num = mol.GetNumConformers()
            assert conf_num == (eq_confs + 1)
            if eq_confs:
                weights = F.softmax(-torch.tensor(energy_lst))
                # random pick one
                pick_lst = [idx for idx in range(conf_num)]
                p_idx = random.choice(pick_lst)

                for conf_id in range(conf_num):
                    if conf_id != p_idx:
                        mol.RemoveConformer(conf_id)
                # only left p_idx
                if p_idx == 0:
                    weight = 1
                else:
                    if self.eq_weight:
                        weight = 1
                    else:
                        weight = weights[p_idx - 1].item()

            else:
                weight = 1

        else:
            ky = str(idx.item()).encode()
            serialized_data = MOL_LST.begin().get(ky)
            mol = pickle.loads(serialized_data)
            # mol = MOL_LST[idx.item()]


        atom_num = mol.GetNumAtoms()

        # get rotate bond  可旋转键识别：构象变化的基础
        if self.addh:
            rotable_bonds = get_torsions([mol])
        else:
            no_h_mol = Chem.RemoveHs(mol)
            rotable_bonds = get_torsions([no_h_mol])


        # prob = random.random()
        cod_denoise = self.cod_denoise
        if self.integrate_coord:
            assert not self.cod_denoise
            prob = random.random()
            if prob < 0.5:
                cod_denoise = True
            else:
                cod_denoise = False


#刚性分子 跳过加噪
        if atom_num != org_atom_num or len(rotable_bonds) == 0 or cod_denoise: # or prob < self.random_pos_prb:
            pos_noise_coords = self.transform_noise(org_data.pos, self.position_noise_scale)
            org_data.pos_target = torch.tensor(pos_noise_coords - org_data.pos.numpy())
            org_data.pos = torch.tensor(pos_noise_coords)


            if self.add_radius_edge: # mimic the painn
                radius_edge_index = radius_graph(org_data.pos, r=self.radius, loop=False)
                org_data.radius_edge_index = radius_edge_index

            if self.equilibrium:
                org_data.w1 = weight
                org_data.wg = torch.tensor([weight for _ in range(org_atom_num)], dtype=torch.float32)
            return org_data

        # else angel random
        # if len(rotable_bonds):
        org_angle = []
        if self.decay:
            rotate_bonds_order, rb_depth = get_rotate_order_info(mol, rotable_bonds)
            decay_coe_lst = []
            for i, rot_bond in enumerate(rotate_bonds_order):
                org_angle.append(GetDihedral(mol.GetConformer(), rot_bond))
                decay_scale = (self.decay_coe) ** (rb_depth[i] - 1)
                decay_coe_lst.append(self.dihedral_angle_noise_scale*decay_scale)
            noise_angle = self.transform_noise_decay(org_angle, self.dihedral_angle_noise_scale, decay_coe_lst)
            new_mol = apply_changes(mol, noise_angle, rotate_bonds_order)
        else:
            if self.bat_noise:
                new_mol, bond_label_lst, angle_label_lst, dihedral_label_lst, rotate_dihedral_label_lst, specific_var_lst = add_equi_noise_new(mol, add_ring_noise=False)
            else:
                for rot_bond in rotable_bonds:
                    org_angle.append(GetDihedral(mol.GetConformer(), rot_bond))
                org_angle = np.array(org_angle)
                noise_angle = self.transform_noise(org_angle, self.dihedral_angle_noise_scale)
                new_mol = apply_changes(mol, noise_angle, rotable_bonds)

        coord_conf = new_mol.GetConformer()
        pos_noise_coords_angle = np.zeros((atom_num, 3), dtype=np.float32)
        # pos_noise_coords = new_mol.GetConformer().GetPositions()
        for idx in range(atom_num):
            c_pos = coord_conf.GetAtomPosition(idx)
            pos_noise_coords_angle[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]

        # coords = np.zeros((atom_num, 3), dtype=np.float32)
        # coord_conf = mol.GetConformer()
        # for idx in range(atom_num):
        #     c_pos = coord_conf.GetAtomPosition(idx)
        #     coords[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]
        # coords = mol.GetConformer().GetPositions()

        if self.bat_noise:
            # check nan
            if torch.tensor(pos_noise_coords_angle).isnan().sum().item():# contains nan
                print('--------bat nan, revert back to org coord-----------')
                pos_noise_coords_angle = org_data.pos.numpy()


        #二次高斯噪声叠加
        pos_noise_coords = self.transform_noise(pos_noise_coords_angle, self.position_noise_scale)
        
        
        # if self.composition or not len(rotable_bonds):
        #     pos_noise_coords = self.transform_noise(coords, self.position_noise_scale)
        #     if len(rotable_bonds): # set coords into the mol
        #         conf = mol.GetConformer()
        #         for i in range(mol.GetNumAtoms()):
        #             x,y,z = pos_noise_coords[i]
        #             conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        

        

        #  只预测高斯噪声部分
        # org_data.pos_target = torch.tensor(pos_noise_coords - org_data.pos.numpy())
        if self.composition:
            org_data.pos_target = torch.tensor(pos_noise_coords - pos_noise_coords_angle)
            org_data.pos = torch.tensor(pos_noise_coords)
        else:
            # denoise angle + guassian noise
            # org_data.pos_target = torch.tensor(pos_noise_coords - org_data.pos.numpy())
            # org_data.pos = torch.tensor(pos_noise_coords)
            
            # only denoise angle
            org_data.pos_target = torch.tensor(pos_noise_coords_angle - org_data.pos.numpy())
            org_data.pos = torch.tensor(pos_noise_coords_angle)
        
        if self.equilibrium:
            org_data.w1 = weight
            org_data.wg = torch.tensor([weight for _ in range(atom_num)], dtype=torch.float32)

        if self.add_radius_edge: # mimic the painn
            radius_edge_index = radius_graph(org_data.pos, r=self.radius, loop=False)
            org_data.radius_edge_index = radius_edge_index
        
        return org_data


#root = r'D:\project\FradNMI-main\torchmdnet\datasets\PCQM4MV2_XYZ'  # 指向 1 里那个文件夹
#dataset = PCQM4MV2_XYZ(root)  # 自动解压 → 递归扫 *.xyz → 生成 .pt
#print(' molecules:', len(dataset))