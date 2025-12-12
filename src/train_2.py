import os
import time
from collections import defaultdict
import torch
import torch.optim as optim

# from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import pickle

from data_loader_pickle import PickleEventGraphDataset
from HRGCN import HRGCN

import boto3


class Train(object):
    def __init__(
        self,
        data_path,
        model_path,
        train_iter_n,
        num_train,
        batch_s,
        mini_batch_s,
        lr,
        save_model_freq,
        s3_stage,
        s3_bucket,
        model_version,
        dataset_id,
        ignore_weight=False,
        source_types=None,
        input_type="single",
        s3_prefix=None,
        sampling_size=None,
        eval_size=None,
        augmentation_method=None,
        insertion_iteration=None,
        subgraph_ratio=None,
        swap_node_pct=None,
        swap_edge_pct=None,
        add_method=None,
        edge_addition_pct=None,
        replace_edges=None,
        edge_mutate_prob=None,
        test_set=True,
        fix_center=True,
        num_eval=None,
        unzip=False,
        split_data=True,
        edge_ratio_percentile=0.95,
        main_loss=None,
        tolerance=None,
        # known_abnormal_ratio=None,
        job_prefix=None,
        **kwargs,
    ):
        super().__init__()

        random.seed(kwargs["random_seed"])
        np.random.seed(kwargs["random_seed"])
        torch.manual_seed(kwargs["random_seed"])
        torch.cuda.manual_seed(kwargs["random_seed"])
        torch.cuda.manual_seed_all(kwargs["random_seed"])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.data_root_dir = data_path
        self.model_path = model_path
        os.makedirs(self.model_path, exist_ok=True)
        self.model_version = model_version
        self.dataset_id = dataset_id
        self.test_set = test_set
        self.split_data = split_data
        self.input_type = input_type

        self.sampling_size = sampling_size
        self.eval_size = eval_size

        self.source_types = None
        self.main_loss = main_loss

        self.augmented = False if kwargs["weighted_loss"] == "ignore" else True

        if source_types is not None:
            self.source_types = [int(i) for i in source_types.split(",")]

        self.fix_center = fix_center

        if self.dataset_id == 2:
            self.pickle_path = kwargs["pickle_path"]
            self.dataset = PickleEventGraphDataset(pickle_path=self.pickle_path, split_data=self.split_data)
            self.graph = self.dataset[0].to(self.device)

        self.num_train_benign = num_train
        self.num_eval = num_eval

        self.embed_d = kwargs["feature_size"]
        self.out_embed_d = kwargs["out_embed_s"]

        self.train_iter_n = train_iter_n
        self.lr = lr

        self.batch_s = batch_s
        self.mini_batch_s = mini_batch_s

        self.save_model_freq = save_model_freq
        self.s3_bucket = s3_bucket
        self.s3_prefix = f"application/anomaly_detection/deeptralog/HetGNN/experiments/model{model_version}_{job_prefix}_{main_loss}_{kwargs['weighted_loss']}_lossweight{kwargs['loss_weight']}_emp{edge_mutate_prob}_eap{edge_addition_pct}_snpct{swap_node_pct}_sepct{swap_edge_pct}_re{replace_edges}_{kwargs['hidden_channels']}_{kwargs['num_hidden_conv_layers']}_/"
        self.s3_stage = s3_stage

        self.model = HRGCN(graph=self.graph, hidden_channels=kwargs['hidden_channels'], out_channels=self.out_embed_d).to(self.device)

        self.parameters = self.model.parameters()
        self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optim, milestones=[60, 120], gamma=0.1
        )

        self.early_stopping = EarlyStopping(tolerance=tolerance)

    def train(self):
        """
        model training method
        """
        print("model training ...")
        self.model.train()

        train_mask, val_mask, test_mask = self.train_eval_test_split()

        # Initialize SVDD center
        with torch.no_grad():
            initial_embeddings = self.model(self.graph.x_dict, self.graph.edge_index_dict)
            train_embeds = initial_embeddings['user'][train_mask]
            svdd_center = torch.mean(train_embeds, dim=0)
            self.model.set_svdd_center(svdd_center)

        for epoch in range(self.train_iter_n):
            self.model.train()
            self.optim.zero_grad()
            
            out = self.model(self.graph.x_dict, self.graph.edge_index_dict)
            train_embeds = out['user'][train_mask]
            
            dist = torch.sum((train_embeds - self.model.get_svdd_center())**2, dim=1)
            loss = torch.mean(dist)
            
            loss.backward()
            self.optim.step()
            
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

            if epoch % self.save_model_freq == 0:
                # Evaluate the model
                print("Evaluating Model on Validation Set..")
                roc_auc, ap = self.eval_model(val_mask)
                
                # Save Model
                torch.save(
                    self.model.state_dict(), f"{self.model_path}/HetGNN_{epoch}.pt"
                )
                if self.s3_stage:
                    self.sync_model_path_to_s3(
                        s3_bucket=self.s3_bucket, s3_prefix=self.s3_prefix
                    )

                self.early_stopping(roc_auc)
                if self.early_stopping.early_stop:
                    print(f"Early Stopping at epoch: {epoch}")
                    break
        
        print("Final Evaluation on Test Set..")
        self.eval_model(test_mask)


    def train_eval_test_split(self):
        """
        splite data into train eval test
        """
        num_nodes = self.graph['user'].num_nodes
        n_nodes = np.arange(num_nodes)
        np.random.shuffle(n_nodes)

        train_size = int(num_nodes * 0.8)
        val_size = int(num_nodes * 0.1)

        train_idx = n_nodes[:train_size]
        val_idx = n_nodes[train_size : train_size + val_size]
        test_idx = n_nodes[train_size + val_size :]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        return train_mask, val_mask, test_mask


    def eval_model(self, mask):
        """
        Eval Model
        """
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.graph.x_dict, self.graph.edge_index_dict)
            eval_embeds = out['user'][mask]
            
            dist = torch.sum((eval_embeds - self.model.get_svdd_center())**2, dim=1)
            
            if hasattr(self.graph['user'], 'y'):
                labels = self.graph['user'].y[mask]
                if labels.numel() == 0:
                    print("\tNo labels for the given mask.")
                    return 0.0, 0.0
                
                fpr, tpr, roc_thresholds = roc_curve(labels.cpu(), dist.cpu())
                roc_auc = auc(fpr, tpr)

                precision, recall, pr_thresholds = precision_recall_curve(
                    labels.cpu(), dist.cpu()
                )
                ap = auc(recall, precision)

                print(f"\tAUC:{roc_auc}; Avg Precision:{ap};")

                return roc_auc, ap
            else:
                print("\tNo 'y' attribute in user nodes for evaluation.")
                # If no labels, just print the average distance
                avg_dist = torch.mean(dist)
                print(f"\tAverage distance from center: {avg_dist:.4f}")

                node_indices = torch.arange(self.graph['user'].num_nodes)[mask]
                scores = dist.cpu().numpy()
                with open("src/anomaly_scores.csv", "w") as f:
                    f.write("node_index,anomaly_score\n")
                    for node_idx, score in zip(node_indices.cpu().numpy(), scores):
                        f.write(f"{node_idx},{score}\n")

                print("\tAnomaly scores saved to anomaly_scores.csv")
                
                return -avg_dist, 0.0

    def sync_model_path_to_s3(self, s3_bucket, s3_prefix):
        """
        sync model path to S3 periodically
        """
        client = boto3.client("s3")

        for root, dirs, files in os.walk(self.model_path):
            for filename in files:
                local_path = os.path.join(root, filename)

                relative_path = os.path.relpath(local_path, self.model_path)

                s3_path = os.path.join(s3_prefix, relative_path)

                try:
                    print(f"Uploading {s3_path}...")
                    client.upload_file(local_path, s3_bucket, s3_path)

                except Exception as e:
                    print(f"Failed to upload {local_path} to {s3_path}.\n{e}")


class EarlyStopping:
    """
    stop if performance exceeds the tolerance
    """

    def __init__(self, tolerance=3):
        self.tolerance = tolerance
        self.counter = 0
        self.previous_score = -1
        self.early_stop = False

    def __call__(self, eval_score):
        if eval_score <= self.previous_score:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.counter = 0
        self.previous_score = eval_score