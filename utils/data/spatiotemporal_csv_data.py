import argparse
import numpy as np
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import utils.data.functions
from utils1 import get_adjacency_matrix_2direction,get_adjacency_matrix


class SpatioTemporalCSVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        feat_path: str,
        adj_path: str,
        batch_size: int = 64,
        seq_len: int = 12,
        pre_len: int = 3,
        split_ratio: float = 0.8,
        normalize: bool = True,
        noise=True,
        noise_ratio=0.2,
        noise_sever=1,
        noise_ratio_node=0.2,
        noise_type='gaussian',
        noise_ratio_test=0.2,
        noise_ratio_node_test=0.2,
        noise_test=True,
        args=None,
        **kwargs
    ):
        super(SpatioTemporalCSVDataModule, self).__init__()
        self.noise=noise
        self.noise_test=noise_test
        self.noise_ratio_test=noise_ratio_test
        self.noise_ratio_node_test=noise_ratio_node_test
        self.noise_ratio=noise_ratio
        self.noise_sever=noise_sever
        self.noise_ratio_node=noise_ratio_node
        self.noise_type=noise_type
        self._feat_path = feat_path
        self._adj_path = adj_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_ratio = split_ratio
        self.normalize = normalize
        self._feat = utils.data.functions.load_features(self._feat_path)
        self._feat_max_val = np.max(self._feat)
        
        self.data_type=args.data
        direction=args.direction
        adj_filename=self._adj_path
        id_filename=args.id_filename

        if self.data_type=='gem':
            num_of_vertices=self._feat.shape[1]
            if direction == 2:
                adj_mx, distance_mx = get_adjacency_matrix_2direction(adj_filename, num_of_vertices, id_filename)
            if direction == 1:
                adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
        else:
            self._adj = utils.data.functions.load_adjacency_matrix(self._adj_path)

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--seq_len", type=int, default=12)
        #parser.add_argument("--adj_path", type=str, default="./")
        parser.add_argument("--pre_len", type=int, default=3)
        parser.add_argument("--split_ratio", type=float, default=0.8)
        parser.add_argument("--normalize", type=bool, default=True)
        parser.add_argument("--id_filename",type=str,default="")
        return parser

    def setup(self, stage: str = None):
        (
            self.train_dataset,
            self.val_dataset,
        ) = utils.data.functions.generate_torch_datasets(
            self._feat,
            self.seq_len,
            self.pre_len,
            split_ratio=self.split_ratio,
            normalize=self.normalize,
            noise=self.noise,
            noise_ratio=self.noise_ratio,
            noise_sever=self.noise_sever,
            noise_ratio_node=self.noise_ratio_node,
            noise_type=self.noise_type,
            noise_ratio_test=self.noise_ratio_test,
            noise_ratio_node_test=self.noise_ratio_node_test,
            noise_test=self.noise_test,
            data_type=self.data_type
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset))

    @property
    def feat_max_val(self):
        return self._feat_max_val

    @property
    def adj(self):
        return self._adj
