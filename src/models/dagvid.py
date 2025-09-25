import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from common.abstract_recommender import GeneralRecommender
from scipy.stats import alpha
from six import text_type
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph
from collections import defaultdict
import json


class DAGVID(GeneralRecommender):
    def __init__(self, config, dataset):
        super(DAGVID, self).__init__(config, dataset)
        self.sparse = True
        self.tau = config['tau']
        self.inter_loss = config['inter_loss']
        self.align_loss = config['align_loss']
        self.exp_loss = config['exp_loss']
        self.cal_loss = config['cal_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.re_loss = config['re_loss']
        self.vt_loss = config['vt_loss']

        # Load dataset information
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.augmented_image_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.augmented_image_user.weight)
        self.augmented_text_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.augmented_text_user.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.dataset_path = dataset_path

        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        user_image_adj_file = os.path.join(dataset_path, 'user_image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        user_text_adj_file = os.path.join(dataset_path, 'user_text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        # Create linear layers for the VIB module
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            # VIB linear layers for the image modality
            self.image_mu_encoder = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
            self.image_logvar_encoder = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
            nn.init.xavier_uniform_(self.image_mu_encoder.weight)
            nn.init.xavier_uniform_(self.image_logvar_encoder.weight)
            nn.init.zeros_(self.image_mu_encoder.bias)
            nn.init.zeros_(self.image_logvar_encoder.bias)

            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            # VIB linear layers for the text modality
            self.text_mu_encoder = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
            self.text_logvar_encoder = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
            nn.init.xavier_uniform_(self.text_mu_encoder.weight)
            nn.init.xavier_uniform_(self.text_logvar_encoder.weight)
            nn.init.zeros_(self.text_mu_encoder.bias)
            nn.init.zeros_(self.text_logvar_encoder.bias)

            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        # user-image
        if os.path.exists(user_image_adj_file):
            self.user_image_adj = torch.load(user_image_adj_file)
        else:
            # First obtain the initial relation matrix for the first aggregation to build user modality similarity
            self.norm_adj = self.get_adj_mat1()
            self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
            # Start constructing
            image_item_embeds = torch.mm(self.image_original_adj, self.image_embedding.weight.detach())
            image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
            user_image_adj = build_sim(image_user_embeds.detach())

            self.user_image_adj = build_knn_normalized_graph(user_image_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                             norm_type='sym')
            torch.save(self.user_image_adj, user_image_adj_file)

        # user-text
        if os.path.exists(user_text_adj_file):
            self.user_text_adj = torch.load(user_text_adj_file)
        else:
            text_item_embeds = torch.mm(self.text_original_adj, self.text_embedding.weight.detach())
            text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
            user_text_adj = build_sim(text_user_embeds.detach())

            self.user_text_adj = build_knn_normalized_graph(user_text_adj, topk=self.knn_k,
                                                            is_sparse=self.sparse,
                                                            norm_type='sym')
            torch.save(self.user_text_adj, user_text_adj_file)

        self.user_inter = self.compute_adj_intersection(self.user_image_adj, self.user_text_adj, 'user')
        self.uu_adj = self.construct_adj_from_intersection(self.user_inter, 'user')
        self.inter = self.compute_adj_intersection(self.image_original_adj, self.text_original_adj, 'item')
        self.ii_adj = self.construct_adj_from_intersection(self.inter, 'item')
        # Component-1 IGA
        self.norm_adj = self.get_adj_mat(self.ii_adj.tolil(), self.uu_adj.tolil())
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        self.softmax = nn.Softmax(dim=-1)

        self.v_filter = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.t_filter = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

    # Forward propagation of the VIB module
    def _vib_forward(self, x, mu_encoder, logvar_encoder, training=True):
        mu = mu_encoder(x)
        logvar = logvar_encoder(x)

        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std, mu, logvar
        else:
            return mu, mu, logvar

    # Calculate adjacency matrix intersections and return intersection dictionary
    def compute_adj_intersection(self, image_adj, text_adj, name):
        inter_file = os.path.join(self.dataset_path, f"{name}_inter.json")
        # Load directly if file exists
        if os.path.exists(inter_file):
            with open(inter_file, 'r', encoding='utf-8') as f:
                adj_intersection = json.load(f)
            return adj_intersection

        # Initialize intersection dictionary and temporary storage lists
        adj_intersection = defaultdict(list)
        img_neighbors = []
        txt_neighbors = []
        adj_length = len(image_adj._indices()[0])

        # Iterate through adjacency matrix indices
        for i in range(adj_length):
            # Get current node ID (ensure main nodes of image and text adjacency matrices are consistent)
            current_id = image_adj._indices()[0][i].item()
            assert current_id == text_adj._indices()[0][i].item(), "Image and text adjacency nodes mismatch"

            # Collect image/text neighbor nodes of the current node
            img_neighbors.append(image_adj._indices()[1][i].item())
            txt_neighbors.append(text_adj._indices()[1][i].item())

            # Calculate intersection after collecting 10 neighbor nodes each
            if len(img_neighbors) == 10 and len(txt_neighbors) == 10:
                # Find intersection of two neighborhoods, excluding self-node
                common_neighbors = list(set(img_neighbors) & set(txt_neighbors))
                common_neighbors = [node for node in common_neighbors if node != current_id]
                adj_intersection[current_id] = common_neighbors

                # Reset temporary lists
                img_neighbors = []
                txt_neighbors = []

        # Save intersection results to file
        with open(inter_file, 'w', encoding='utf-8') as f:
            json.dump(adj_intersection, f)

        return adj_intersection

    # Construct adjacency matrix from intersection results
    def construct_adj_from_intersection(self, adj_intersection, name):
        # Collect row and column indices of the adjacency matrix
        adj_rows = []
        adj_cols = []

        # Iterate through intersection dictionary to populate indices
        for main_node, neighbor_nodes in adj_intersection.items():
            if not neighbor_nodes:  # Skip if no neighbor nodes
                continue
            # Add an edge for each neighbor node (main_node → neighbor)
            for neighbor in neighbor_nodes:
                adj_rows.append(int(main_node))
                adj_cols.append(neighbor)

        # Construct adjacency matrix values (all 1s, indicating existing edges)
        adj_values = torch.ones(len(adj_rows), dtype=torch.int).tolist()

        # Determine adjacency matrix shape based on node type (user/item)
        if name == 'item':
            adj_shape = (self.n_items, self.n_items)
        else:
            adj_shape = (self.n_users, self.n_users)

        # Construct COO-format sparse adjacency matrix
        intersection_adj = sp.coo_matrix(
            (adj_values, (adj_rows, adj_cols)),
            shape=adj_shape,
            dtype=np.int32
        )
        return intersection_adj

    def pre_epoch_processing(self):
        pass

    def get_adj_mat1(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat.tocsr()

    def get_adj_mat(self, item_adj, user_adj):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T

        adj_mat[self.n_users:, self.n_users:] = item_adj
        adj_mat[:self.n_users, :self.n_users] = user_adj

        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()

        self.R = norm_adj_mat[:self.n_users, self.n_users:]

        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def sq_sum(self, emb):
        return 1. / 2 * (emb ** 2).sum()

    # User-item graph convolution (multi-layer aggregation + averaging)
    def user_item_graph_convolution(self, adj, user_embeds, item_embeds):
        # Concatenate initial embeddings of users and items
        current_emb = torch.cat([user_embeds, item_embeds], dim=0)
        # Store aggregation results of each layer
        layer_embeddings = [current_emb]

        # Multi-layer graph convolution aggregation
        for _ in range(self.n_ui_layers):
            # Sparse matrix multiplication for graph convolution
            current_emb = torch.sparse.mm(adj, current_emb)
            # Save current layer results
            layer_embeddings.append(current_emb)

        # Average results from all layers
        layer_embeddings = torch.stack(layer_embeddings, dim=1)
        avg_embeddings = layer_embeddings.mean(dim=1, keepdim=False)
        return avg_embeddings

    # Item-item graph convolution (multi-layer aggregation, returns last layer)
    def item_item_graph_convolution(self, ii_adj, modal_embeds):
        # Initialize current embedding as input modal embedding
        current_modal_emb = modal_embeds.clone()
        # Multi-layer graph convolution aggregation
        for _ in range(self.n_layers):
            # Update embedding with sparse matrix multiplication
            current_modal_emb = torch.sparse.mm(ii_adj, current_modal_emb)
        # Return last layer aggregation result
        return current_modal_emb

    def forward(self, adj, train, v_embeds=None, t_embeds=None):
        # Process image and text features using the fused VIB module
        if self.v_feat is not None:
            image_feats, image_mu, image_logvar = self._vib_forward(
                self.image_embedding.weight,
                self.image_mu_encoder,
                self.image_logvar_encoder,
                train
            )
        else:
            image_feats = image_mu = image_logvar = None

        if self.t_feat is not None:
            text_feats, text_mu, text_logvar = self._vib_forward(
                self.text_embedding.weight,
                self.text_mu_encoder,
                self.text_logvar_encoder,
                train
            )
        else:
            text_feats = text_mu = text_logvar = None

        # Initial filtering noise (with ID embeddings)
        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.v_filter(image_feats))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.t_filter(text_feats))

        # User-Item View
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        augmented_id_embeds = self.user_item_graph_convolution(adj, user_embeds, item_embeds)
        content_embeds = augmented_id_embeds

        # Image
        core_image_item = self.item_item_graph_convolution(self.image_original_adj, image_item_embeds)
        core_image_user = torch.sparse.mm(self.R, core_image_item)
        core_image_embeds = torch.cat([core_image_user, core_image_item], dim=0)

        augmented_image_embeds = self.user_item_graph_convolution(adj, self.augmented_image_user.weight,
                                                                 core_image_item)
        # Text
        core_text_item = self.item_item_graph_convolution(self.text_original_adj, text_item_embeds)
        core_text_user = torch.sparse.mm(self.R, core_text_item)
        core_text_embeds = torch.cat([core_text_user, core_text_item], dim=0)

        augmented_text_embeds = self.user_item_graph_convolution(adj, self.augmented_text_user.weight,
                                                                core_text_item)

        augmented_it_embeds = (augmented_image_embeds + augmented_text_embeds) / 2

        # Attention Fusion
        att_common = torch.cat([self.attention(core_image_embeds), self.attention(core_text_embeds)], dim=-1)
        weight_common = self.softmax(att_common)
        common_embeds = weight_common[:, 0].unsqueeze(dim=1) * core_image_embeds + weight_common[:, 1].unsqueeze(
            dim=1) * core_text_embeds

        side_embeds = (core_image_embeds + core_text_embeds - common_embeds) / 3

        all_embeds = content_embeds + side_embeds

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds, augmented_it_embeds, core_image_embeds, core_text_embeds, image_mu, image_logvar, text_mu, text_logvar, image_feats, text_feats

        return all_embeddings_users, all_embeddings_items, core_image_embeds, core_text_embeds

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def kl_div(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds, augmented_it_embeds, core_image_embeds, core_text_embeds, image_mu, image_logvar, text_mu, text_logvar, image_feats, text_feats = self.forward(
            self.norm_adj, True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        augmented_it_user, augmented_it_items = torch.split(augmented_it_embeds, [self.n_users, self.n_items], dim=0)

        bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)

        # component-2  - RMAR
        # - EXP
        exp_loss = self.InfoNCE(augmented_it_user[users], side_embeds_users[users], self.tau)
        reg_loss = self.sq_sum(augmented_it_items[pos_items]) / self.batch_size

        _, image_item_embeds = torch.split(core_image_embeds, [self.n_users, self.n_items], dim=0)
        _, text_item_embeds = torch.split(core_text_embeds, [self.n_users, self.n_items], dim=0)

        pos_image_item_embeds = image_item_embeds[pos_items]
        pos_text_item_embeds = text_item_embeds[pos_items]
        neg_image_item_embeds = image_item_embeds[neg_items]
        neg_text_item_embeds = text_item_embeds[neg_items]

        # - CAL
        imagemodal_loss = self.bpr_loss(u_g_embeddings, pos_image_item_embeds, neg_image_item_embeds)
        textmodal_loss = self.bpr_loss(u_g_embeddings, pos_text_item_embeds, neg_text_item_embeds)
        cal_loss = imagemodal_loss + textmodal_loss

        # component-3  VID.
        # align
        align_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], self.tau) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], self.tau)
        # inter
        inter_loss = self.InfoNCE(u_g_embeddings, content_embeds_items[pos_items], self.tau) + self.InfoNCE(
            u_g_embeddings, side_embeds_items[pos_items], self.tau)

        # re_loss
        re_loss = 0.0
        if image_mu is not None and image_logvar is not None:
            re_loss += self.kl_div(image_mu, image_logvar)
        if text_mu is not None and text_logvar is not None:
            re_loss += self.kl_div(text_mu, text_logvar)
        
        # vt_loss
        if image_feats is not None and text_feats is not None:
            image_feats = image_feats[pos_items]
            text_feats = text_feats[pos_items]

            vt_loss = self.InfoNCE(image_feats, text_feats, self.tau)
            vt_loss += self.InfoNCE(text_feats, image_feats, self.tau)
        else:
            vt_loss = torch.tensor(0.0, device=self.device)

        

        return bpr_loss + self.align_loss * align_loss + self.inter_loss * inter_loss + self.exp_loss * exp_loss + self.reg_weight * reg_loss + self.cal_loss * cal_loss + self.re_loss * re_loss + self.vt_loss * vt_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e, _, _ = self.forward(self.norm_adj, False)
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
