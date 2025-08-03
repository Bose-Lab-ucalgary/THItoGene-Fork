import glob
import os

import anndata as ad
from anndata.io import read_h5ad as read_h5ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scprep as scp
import torch
import torchvision.transforms as transforms
from PIL import ImageFile, Image
from pathlib import Path

from graph_construction import calcADJ

Image.MAX_IMAGE_PIXELS = 2300000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ViT_HER2ST(torch.utils.data.Dataset):

    def __init__(self, train=True, gene_list_source=None, ds=None, sr=False, fold=0):
        super(ViT_HER2ST, self).__init__()
        self.cnt_dir = r'./data/her2st/data/ST-cnts'
        self.img_dir = r'./data/her2st/data/ST-imgs'
        self.pos_dir = r'./data/her2st/data/ST-spotfiles'
        self.lbl_dir = r'./data/her2st/data/ST-pat/lbl'

        self.r = 224 // 4

        gene_list_source = list(np.load(r'./data/her_hvg_cut_1000.npy', allow_pickle=True))

        self.gene_list_source = gene_list_source

        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]

        self.train = train
        self.sr = sr

        samples = names[1:33]

        te_names = [samples[fold]]

        tr_names = list(set(samples) - set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in self.names}

        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in self.names}

        self.label = {i: None for i in self.names}

        self.lbl2id = {'invasive cancer': 0, 'breast glands': 1, 'immune infiltrate': 2, 'cancer in situ': 3,
                       'connective tissue': 4, 'adipose tissue': 5, 'undetermined': -1}

        if not train and self.names[0] in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']:
            self.lbl_dict = {i: self.get_lbl(i) for i in self.names}
            idx = self.meta_dict[self.names[0]].index
            lbl = self.lbl_dict[self.names[0]]
            lbl = lbl.loc[idx, :]['label'].values
            self.label[self.names[0]] = lbl
        elif train:
            for i in self.names:
                idx = self.meta_dict[i].index
                if i in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']:
                    lbl = self.get_lbl(i)
                    lbl = lbl.loc[idx, :]['label'].values
                    lbl = torch.Tensor(list(map(lambda i: self.lbl2id[i], lbl)))
                    self.label[i] = lbl
                else:
                    self.label[i] = torch.full((len(idx),), -1)

        self.gene_set = list(gene_list_source)
        self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i, m in
                         self.meta_dict.items()}
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                            self.meta_dict.items()}
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}
        self.patch_dict = {}
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.transforms = transforms.Compose([transforms.ColorJitter(0.5, 0.5, 0.5), transforms.RandomHorizontalFlip(),
                                              transforms.RandomRotation(degrees=180), transforms.ToTensor()])
        self.adj_dict = {i: calcADJ(coord=m, k=4, pruneTag='NA') for i, m in self.loc_dict.items()}

    def filter_helper(self):
        a = np.zeros(len(self.gene_list_source))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list_source))):
                a[j] += np.sum(exp[:, j])

    def __getitem__(self, index):
        i = index
        name = self.id2name[i]
        im = self.img_dict[self.id2name[i]]
        im = im.permute(1, 0, 2)
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        label = self.label[self.id2name[i]]
        if self.id2name[i] in self.patch_dict:
            patches = self.patch_dict[self.id2name[i]]
        else:
            patches = None

        adj = self.adj_dict[name]

        patch_dim = 3 * self.r * self.r * 4

        if self.sr:
            centers = torch.LongTensor(centers)

            max_x = centers[:, 0].max().item()
            max_y = centers[:, 1].max().item()
            min_x = centers[:, 0].min().item()
            min_y = centers[:, 1].min().item()
            r_x = (max_x - min_x) // 30
            r_y = (max_y - min_y) // 30

            centers = torch.LongTensor([min_x, min_y]).view(1, -1)
            positions = torch.LongTensor([0, 0]).view(1, -1)
            x = min_x
            y = min_y

            while y < max_y:
                x = min_x
                while x < max_x:
                    centers = torch.cat((centers, torch.LongTensor([x, y]).view(1, -1)), dim=0)
                    positions = torch.cat((positions, torch.LongTensor([x // r_x, y // r_y]).view(1, -1)), dim=0)
                    x += 56
                y += 56

            centers = centers[1:, :]
            positions = positions[1:, :]

            n_patches = len(centers)
            patches = torch.zeros((n_patches, patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                patches[i] = patch.flatten()
            return patches, positions, centers

        else:
            n_patches = len(centers)
            exps = torch.Tensor(exps)
            if patches is None:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))
                for i in range(n_patches):
                    center = centers[i]
                    x, y = center
                    patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                    patches[i] = patch.permute(2, 0, 1)
                self.patch_dict[name] = patches
            if self.train:
                return patches, positions, exps, adj
            else:
                return patches, positions, exps, torch.Tensor(centers), adj

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name
        print(path)
        im = Image.open(path)
        return im

    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '.tsv'
        df = pd.read_csv(path, sep='\t', index_col=0)
        return df

    def get_pos(self, name):
        path = self.pos_dir + '/' + name + '_selection.tsv'
        df = pd.read_csv(path, sep='\t')
        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        return df

    def get_lbl(self, name):
        path = self.lbl_dir + '/' + name + '_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')
        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)

        df.set_index('id', inplace=True)
        return df

    def get_meta(self, name, gene_list_source=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        return meta

    def get_overlap(self, meta_dict, gene_list_source):
        gene_set = set(gene_list_source)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)

class ViT_HEST1K(torch.utils.data.Dataset):
    def __init__(self, mode='train', gene_list = '3CA', sr=False, neighs=4):
        super().__init__()
        
        self.hest_path = Path("/work/bose_lab/tahsin/data/HEST")
        self.patch_path = Path("../../../data/HERST_preprocess/patches_112x112") # Same patch size as her2st?
        self.processed_path = Path("../../data/HERST_preprocess")
        self.r = 224//4  # Same as ViT_HER2ST        
        self.gene_list_source = gene_list
        self.mode = mode
        self.sr = sr
        self.prune='NA'
        self.neighs = neighs
        
        if gene_list == "HER2ST":
            self.processed_path = self.processed_path / 'HER2ST'
        elif gene_list == "cSCC":
            self.processed_path = self.processed_path / 'cSCC'
        elif gene_list == "3CA":
            self.processed_path = self.processed_path / '3CA_genes'
        elif gene_list == "Hallmark":
            self.processed_path = self.processed_path / 'Hallmark_genes'

        if mode == 'train':
            self.processed_path = self.processed_path / 'train'
        elif mode == 'val':
            self.processed_path = self.processed_path / 'val'
        else:
            self.processed_path = self.processed_path / 'test'
        
        print(f"Looking for HEST1K data in: {self.processed_path}")

        # Get sample IDs from available files
        if self.processed_path.exists():
            sample_files = list(self.processed_path.glob("*.h5ad"))
            self.sample_ids = [file.stem.split('_')[0] for file in sample_files]
            print(f"Found {len(self.sample_ids)} samples.")
        else:
            print(f"Warning: Path {self.processed_path} does not exist.")
            raise FileNotFoundError(f"Processed path {self.processed_path} not found.")
            # self.sample_ids = ['dummy_sample']
        self.id2name = dict(enumerate(self.sample_ids))
        
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        print(f"\nProcessing sample {sample_id}")
        adata_path = os.path.join(self.processed_path, f"{sample_id}_preprocessed.h5ad")
        try:
            adata = read_h5ad(adata_path)   
        except Exception as e:
            print(f"Error reading {adata_path}: {e}")
            raise Exception(f"Failed to load data for sample {sample_id}. Check if the file exists and is valid.")
        
        # Make var_names unique before any indexing
        if not adata.var_names.is_unique:
            print(f"Found {sum(adata.var_names.duplicated())} duplicate gene names, making them unique")
            # Method 1: Make unique by appending _1, _2, etc. to duplicates
            adata.var_names_make_unique()

        exps = adata.X
        
        # Get array coordinates
        if 'array_row' in adata.obs and 'array_col' in adata.obs:
            loc = adata.obs[['array_row', 'array_col']].values.astype(int)
        elif 'spatial' in adata.obsm:
            loc = adata.obsm['spatial'].copy()
        else:
            print(f"Error: Sample {sample_id} does not have spatial coordinates.")
            loc = np.zeros((adata.n_obs, 2), dtype=int)  
            for i in range(adata.n_obs):
                loc[i] = [i // 64, i % 64]   
                
        # Normalize positions to [0, 63] range like HER2ST
        pos_min = loc.min(axis=0)
        pos_max = loc.max(axis=0)
        
        # Normalize to [0, 1] then scale to [0, 63]
        loc_normalized = (loc - pos_min) / (pos_max - pos_min + 1e-8)
        loc_scaled = (loc_normalized * 63).astype(int)
        loc_scaled = np.clip(loc_scaled, 0, 63)  # Ensure within bounds
        
        loc = torch.LongTensor(loc_scaled)
    
        # Get pixel coordinates
        if 'spatial' in adata.obsm:
            centers = adata.obsm['spatial']
        elif 'spatial' in adata.uns:
            centers = adata.uns['spatial']
        else:
            # print(f"Error: Sample {sample_id} does not have spatial coordinates in obsm['spatial'].")
            print(adata)
            raise ValueError(f"Sample {sample_id} does not have spatial coordinates in obsm['spatial'] or uns['spatial'].")
        # centers = adata.obsm['spatial']
        
        # Load Patches
        patch_path = os.path.join(self.patch_path, f"{sample_id}.h5")
        if os.path.exists(patch_path):
            patches = self._load_patches(sample_id, adata.obs_names)
        else:
            patches = np.random.randn(len(adata), 3, 112, 112)
        # Get adjacency matrix 
        adj_matrix = calcADJ(loc, self.neighs, pruneTag=self.prune)
            
        # print(f"Patches shape after loading: {patches.shape}")
        patches = torch.FloatTensor(patches)
        # print(f"Patches shape as tensor: {patches.shape}")
        
        exps = torch.Tensor(exps)
        centers = torch.FloatTensor(centers)
        loc = torch.LongTensor(loc)
        adj_matrix = torch.FloatTensor(adj_matrix) if adj_matrix is not None else None


        if self.mode == 'train' or self.mode == 'val':
            return patches, loc, exps, adj_matrix
        else:
            return patches, loc, exps, centers, adj_matrix
        
    def _load_patches(self, sample_id, spot_names):
        patches = []
        path = os.path.join(self.patch_path, f"{sample_id}.h5")
        
        with h5py.File(path, 'r') as f:
            images = f['img'][:]
            barcodes = [bc[0].decode('utf-8') if isinstance(bc[0], bytes) else bc[0] for bc in f['barcode'][:]]
            
            barcode_to_idx = {bc: i for i, bc in enumerate(barcodes)}
            
            
            for spot in spot_names:
                if spot in barcode_to_idx:
                    idx = barcode_to_idx[spot]
                    img = images[idx]
                    # patches.append(images[idx])

                    # Why convert to tensor and normalize??
                    if len(img.shape) == 2:
                        img = np.stack([img, img, img], axis =0) # Convert grayscale to RGB
                    else:
                        img = img.transpose(2, 0, 1) # Convert HxWxC to CxHxW
                    patches.append(img)
                else:
                    patches.append(np.zeros((3, 112, 112)))
                    
        return np.array(patches)
    
    def __len__(self):
        return len(self.sample_ids)
    
    def get_gene_names(self):
        """Get gene names from the first sample"""
        if len(self.sample_ids) > 0:
            adata_path = os.path.join(self.processed_path, f"{self.sample_ids[0]}_preprocessed.h5ad")
            adata = ad.read_h5ad(adata_path)
            return adata.var_names.tolist()
        else:
            return []

class ViT_SKIN(torch.utils.data.Dataset):

    def __init__(self, train=True, gene_list_source=None, ds=None, sr=False, aug=False, norm=False, fold=0):
        super(ViT_SKIN, self).__init__()
        self.dir = './data/GSE144240_RAW'
        self.r = 224 // 4
        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i + '_ST_' + j)

        gene_list_source = list(
            np.load('./data/skin_hvg_cut_1000.npy', allow_pickle=True))

        self.gene_list_source = gene_list_source

        self.train = train
        self.sr = sr
        self.aug = aug
        self.transforms = transforms.Compose([transforms.ColorJitter(0.5, 0.5, 0.5), transforms.ToTensor()])
        self.norm = norm

        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))

        if train:
            names = tr_names
        else:
            names = te_names

        print('Loading imgs...')
        if self.aug:
            self.img_dict = {i: self.get_img(i) for i in names}
        else:
            self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in names}

        self.gene_set = list(gene_list_source)

        if self.norm:
            self.exp_dict = {
                i: sc.pp.scale(scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))) for
                i, m in self.meta_dict.items()}
        else:
            self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for
                             i, m in self.meta_dict.items()}
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                            self.meta_dict.items()}
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.patch_dict = {}

        self.adj_dict = {i: calcADJ(coord=m, k=4, pruneTag='NA') for i, m in self.loc_dict.items()}

    def filter_helper(self):
        a = np.zeros(len(self.gene_list_source))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list_source))):
                a[j] += np.sum(exp[:, j])

    def __getitem__(self, index):
        i = index
        name = self.id2name[i]
        im = self.img_dict[self.id2name[i]]
        if self.aug:
            im = self.transforms(im)
            # im = im.permute(1,2,0)
            im = im.permute(2, 1, 0)
        else:
            im = im.permute(1, 0, 2)  # im = im

        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4

        if self.id2name[i] in self.patch_dict:
            patches = self.patch_dict[self.id2name[i]]
        else:
            patches = None
        adj = self.adj_dict[name]

        if self.sr:
            centers = torch.LongTensor(centers)
            max_x = centers[:, 0].max().item()
            max_y = centers[:, 1].max().item()
            min_x = centers[:, 0].min().item()
            min_y = centers[:, 1].min().item()
            r_x = (max_x - min_x) // 30
            r_y = (max_y - min_y) // 30

            centers = torch.LongTensor([min_x, min_y]).view(1, -1)
            positions = torch.LongTensor([0, 0]).view(1, -1)
            x = min_x
            y = min_y

            while y < max_y:
                x = min_x
                while x < max_x:
                    centers = torch.cat((centers, torch.LongTensor([x, y]).view(1, -1)), dim=0)
                    positions = torch.cat((positions, torch.LongTensor([x // r_x, y // r_y]).view(1, -1)), dim=0)
                    x += 56
                y += 56

            centers = centers[1:, :]
            positions = positions[1:, :]

            n_patches = len(centers)
            patches = torch.zeros((n_patches, patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                patches[i] = patch.flatten()

            return patches, positions, centers

        else:
            n_patches = len(centers)
            exps = torch.Tensor(exps)
            if patches is None:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))

                for i in range(n_patches):
                    center = centers[i]
                    x, y = center
                    patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r),
                            :]
                    patches[i] = patch.permute(2, 0, 1)
                self.patch_dict[name] = patches
            if self.train:
                return patches, positions, exps, adj
            else:
                return patches, positions, exps, torch.Tensor(centers), adj

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        path = glob.glob(self.dir + '/*' + name + '.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self, name):
        path = glob.glob(self.dir + '/*' + name + '_stdata.tsv')[0]
        df = pd.read_csv(path, sep='\t', index_col=0)
        return df

    def get_pos(self, name):

        pattern = f"{self.dir}/*spot*{name}*.tsv"
        path = glob.glob(pattern)[0]
        print(path)
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_meta(self, name, gene_list_source=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'), how='inner')
        return meta

    def get_overlap(self, meta_dict, gene_list_source):
        gene_set = set(gene_list_source)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)


class DATA_BRAIN(torch.utils.data.Dataset):

    def __init__(self, train=True, gene_list_source=None, ds=None, sr=False, aug=False, norm=False, fold=0):
        super(DATA_BRAIN, self).__init__()
        self.dir = './data/10X'
        self.r = 224 // 4

        sample_names = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673',
                        '151674', '151675', '151676']

        gene_list_source = list(np.load('./data/10X/final_gene.npy'))

        self.gene_list_source = gene_list_source

        self.train = train
        self.sr = sr
        self.aug = aug
        self.transforms = transforms.Compose([transforms.ColorJitter(0.5, 0.5, 0.5), transforms.ToTensor()])
        self.norm = norm

        samples = sample_names
        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))

        if train:
            names = tr_names
        else:
            names = te_names

        print('Loading imgs...')
        if self.aug:
            self.img_dict = {i: self.get_img(i) for i in names}
        else:
            self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in names}

        self.gene_set = list(gene_list_source)

        if self.norm:
            self.exp_dict = {
                i: sc.pp.scale(scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))) for
                i, m in self.meta_dict.items()}
        else:
            self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for
                             i, m in self.meta_dict.items()}
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                            self.meta_dict.items()}
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.patch_dict = {}

        self.adj_dict = {i: calcADJ(coord=m, k=4, pruneTag='NA') for i, m in self.loc_dict.items()}

    def filter_helper(self):
        a = np.zeros(len(self.gene_list_source))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list_source))):
                a[j] += np.sum(exp[:, j])

    def __getitem__(self, index):
        i = index
        name = self.id2name[i]
        im = self.img_dict[self.id2name[i]]
        if self.aug:
            im = self.transforms(im)
            im = im.permute(2, 1, 0)
        else:
            im = im.permute(1, 0, 2)  # im = im

        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4

        if self.id2name[i] in self.patch_dict:
            patches = self.patch_dict[self.id2name[i]]
        else:
            patches = None
        adj = self.adj_dict[name]

        if self.sr:
            centers = torch.LongTensor(centers)
            max_x = centers[:, 0].max().item()
            max_y = centers[:, 1].max().item()
            min_x = centers[:, 0].min().item()
            min_y = centers[:, 1].min().item()
            r_x = (max_x - min_x) // 30
            r_y = (max_y - min_y) // 30

            centers = torch.LongTensor([min_x, min_y]).view(1, -1)
            positions = torch.LongTensor([0, 0]).view(1, -1)
            x = min_x
            y = min_y

            while y < max_y:
                x = min_x
                while x < max_x:
                    centers = torch.cat((centers, torch.LongTensor([x, y]).view(1, -1)), dim=0)
                    positions = torch.cat((positions, torch.LongTensor([x // r_x, y // r_y]).view(1, -1)), dim=0)
                    x += 56
                y += 56

            centers = centers[1:, :]
            positions = positions[1:, :]

            n_patches = len(centers)
            patches = torch.zeros((n_patches, patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                patches[i] = patch.flatten()

            return patches, positions, centers

        else:
            n_patches = len(centers)
            exps = torch.Tensor(exps)
            if patches is None:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))

                for i in range(n_patches):
                    center = centers[i]
                    x, y = center
                    patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r),
                            :]
                    patches[i] = patch.permute(2, 0, 1)
                self.patch_dict[name] = patches
            if self.train:
                return patches, positions, exps, adj
            else:
                return patches, positions, exps, torch.Tensor(centers), adj

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        path = glob.glob(self.dir + f'/{name}/{name}_full_image.tif')[0]
        im = Image.open(path)
        return im

    def get_meta(self, name, gene_list_source=None):
        meta = pd.read_csv('./data/10X/151507/10X_Visium_151507_meta.csv', index_col=0)
        return meta

    def get_overlap(self, meta_dict, gene_list_source):
        gene_set = set(gene_list_source)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)


