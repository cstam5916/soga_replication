import os.path as osp
import pandas as pd
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

class PygOgbnProteins(PygNodePropPredDataset):
    def __init__(self, meta_csv = None):
        root, name, transform = '/kaggle/input', 'ogbn-proteins', T.ToSparseTensor()
        if meta_csv is None:
            meta_csv = osp.join(root, name, 'ogbn-master.csv')
        master = pd.read_csv(meta_csv, index_col = 0)
        meta_dict = master[name]
        meta_dict['dir_path'] = osp.join(root, name)
        super().__init__(name = name, root = root, transform = transform, meta_dict = meta_dict)
    def get_idx_split(self, split_type = None):
        if split_type is None:
            split_type = self.meta_info['split']
        path = osp.join(self.root, 'split', split_type)
        if osp.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))
        if self.is_hetero:
            train_idx_dict, valid_idx_dict, test_idx_dict = read_nodesplitidx_split_hetero(path)
            for nodetype in train_idx_dict.keys():
                train_idx_dict[nodetype] = torch.from_numpy(train_idx_dict[nodetype]).to(torch.long)
                valid_idx_dict[nodetype] = torch.from_numpy(valid_idx_dict[nodetype]).to(torch.long)
                test_idx_dict[nodetype] = torch.from_numpy(test_idx_dict[nodetype]).to(torch.long)
                return {'train': train_idx_dict, 'valid': valid_idx_dict, 'test': test_idx_dict}
        else:
            train_idx = dt.fread(osp.join(path, 'train.csv'), header = None).to_numpy().T[0]
            train_idx = torch.from_numpy(train_idx).to(torch.long)
            valid_idx = dt.fread(osp.join(path, 'valid.csv'), header = None).to_numpy().T[0]
            valid_idx = torch.from_numpy(valid_idx).to(torch.long)
            test_idx = dt.fread(osp.join(path, 'test.csv'), header = None).to_numpy().T[0]
            test_idx = torch.from_numpy(test_idx).to(torch.long)
            return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
