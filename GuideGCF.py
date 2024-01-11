import numpy as np
import scipy.sparse as sp
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

import utils
from spmm import SpecialSpmm, CHUNK_SIZE_FOR_SPMM

from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import pairwise_distances


class GuideGCF(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(GuideGCF, self).__init__(config, dataset)

        # load base para
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.sim_user_num = config['sim_user_num']

        self.topk_add = config['topk_add']
        self.topk_del = config['topk_del']
        self.dataset = config['dataset']
        self.store_idx_flag = config['store_idx']
        self.path = './dataset/' + self.dataset
        print('user: add by topk %d ,values by sim; del by topk %d' % (self.topk_add, self.topk_del))

        # generate interaction_matrix
        self.inter_matrix_type = config['inter_matrix_type']
        value_field = self.RATING if self.inter_matrix_type == 'rating' else None
        self.interaction_matrix = dataset.inter_matrix(form='coo', value_field=value_field).astype(
            np.float32)

        # define layers
        self.user_linear = torch.nn.Linear(in_features=self.n_items, out_features=self.embedding_size,
                                           bias=False)
        self.item_linear = torch.nn.Linear(in_features=self.n_users, out_features=self.embedding_size,
                                           bias=False)

        # define loss
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # generate intermediate data
        self.adj_matrix = self.get_adj_mat(self.interaction_matrix)
        self.norm_adj_matrix = self.get_norm_mat(self.adj_matrix).to(self.device)

        # for learn adj
        self.spmm = config['spmm']
        self.special_spmm = SpecialSpmm() if self.spmm == 'spmm' else torch.sparse.mm

        self.prune_threshold = 0.02
        self.MIM_weight = config['MIM_weight']
        self.tau = config['tau']
        self.aug_ratio = config['aug_ratio']
        self.pool_multi = 10

        self.for_learning_adj()

        # Get Project Pool
        self.user_idex_tensor, self.item_idex_tensor = self.store_idx()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(self._init_weights)
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def for_learning_adj(self):
        self.adj_indices = self.norm_adj_matrix.indices()
        self.adj_shape = self.norm_adj_matrix.shape
        self.adj = self.norm_adj_matrix

        inter_data = torch.FloatTensor(self.interaction_matrix.data).to(self.device)
        inter_user = torch.LongTensor(self.interaction_matrix.row).to(self.device)
        inter_item = torch.LongTensor(self.interaction_matrix.col).to(self.device)
        inter_mask = torch.stack([inter_user, inter_item], dim=0)

        self.inter_spTensor = torch.sparse.FloatTensor(inter_mask, inter_data,
                                                       self.interaction_matrix.shape).coalesce()
        self.inter_spTensor_t = self.inter_spTensor.t().coalesce()

        self.inter_indices = self.inter_spTensor.indices()
        self.inter_shape = self.inter_spTensor.shape

    def store_idx(self):
        if self.store_idx_flag:
            print('Load jc_i.pt')
            #jc_i = torch.load(self.path + '/jc_i.pt')
            print('Load user_idex.pt')
            u_list = torch.load(self.path + '/user_idex.pt')
            print('Load item_idex.pt')
            i_list = torch.load(self.path + '/item_idex.pt')
            return u_list,i_list
        else:
            u_list, i_list = self.item_pool()
            return u_list, i_list

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.01)

    # Returns: torch.FloatTensor: The embedding tensor of all user, shape: [n_users, embedding_size]
    def get_all_user_embedding(self):
        all_user_embedding = torch.sparse.mm(self.inter_spTensor, self.user_linear.weight.t())
        return all_user_embedding

    def get_all_item_embedding(self):
        all_item_embedding = torch.sparse.mm(self.inter_spTensor_t, self.item_linear.weight.t())
        return all_item_embedding

    def get_adj_mat(self, inter_M, data=None):
        if data is None:
            data = [1] * inter_M.data
        inter_M_t = inter_M.transpose()
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items),
                          dtype=np.float32)
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), data))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), data)))
        A._update(data_dict)
        return A

    def get_norm_mat(self, A):
        r""" A_{hat} = D^{-0.5} \times A \times D^{-0.5} """
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        SparseL = utils.sp2tensor(L)
        return SparseL

    # Learn adj
    def sp_cos_sim(self, a, b, eps=1e-8, CHUNK_SIZE=CHUNK_SIZE_FOR_SPMM):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))

        L = self.inter_indices.shape[1]
        sims = torch.zeros(L, dtype=a.dtype).to(self.device)
        for idx in range(0, L, CHUNK_SIZE):
            batch_indices = self.inter_indices[:, idx:idx + CHUNK_SIZE]

            a_batch = torch.index_select(a_norm, 0, batch_indices[0, :])
            b_batch = torch.index_select(b_norm, 0, batch_indices[1, :])

            dot_prods = torch.mul(a_batch, b_batch).sum(1)
            sims[idx:idx + CHUNK_SIZE] = dot_prods

        return torch.sparse_coo_tensor(self.inter_indices, sims, size=self.interaction_matrix.shape,
                                       dtype=sims.dtype).coalesce()

    def get_sim_mat(self):
        user_feature = self.get_all_user_embedding().to(self.device)
        item_feature = self.get_all_item_embedding().to(self.device)
        sim_inter = self.sp_cos_sim(user_feature, item_feature)
        return sim_inter

    def inter2adj(self, inter):
        inter_t = inter.t().coalesce()
        data = inter.values()
        data_t = inter_t.values()
        adj_data = torch.cat([data, data_t], dim=0)
        adj = torch.sparse.FloatTensor(self.adj_indices, adj_data, self.adj_shape).to(
            self.device).coalesce()
        return adj

    def item_pool(self):
        stime = time.time()
        U_i_inter_dense = self.interaction_matrix.todense()
        print('creat jc_i.pt')
        user_similar_jaccard = 1 - pairwise_distances(U_i_inter_dense, metric='jaccard')
        #torch.save(user_similar_jaccard, self.path + '/jc_i.pt')
        user_similar_jaccard = pd.DataFrame(user_similar_jaccard)
        etime = time.time()
        print(f'Calculate similarity time: {etime - stime}s')

        stime = time.time()
        topN_users = {}

        for i in user_similar_jaccard.index:
            _df = user_similar_jaccard.loc[i].drop([i])
            _df_sorted = _df.sort_values(ascending=False)
            top_user = list(_df_sorted.index[:self.sim_user_num])
            topN_users[i] = top_user
        sim_users_item_pool = {}
        for i in user_similar_jaccard.index:
            item_pool_rows, item_pool_cols = np.nonzero(U_i_inter_dense[topN_users[i]])
            myself_item_rows, myself_item_pool_cols = np.nonzero(U_i_inter_dense[i])
            item_pool = np.concatenate((item_pool_cols, myself_item_pool_cols))
            sim_users_item_pool[i] = list(set(item_pool))

        histor_users_item_pool = {}
        for i in user_similar_jaccard.index:
            myself_item_rows, myself_item_pool_cols = np.nonzero(U_i_inter_dense[i])
            histor_users_item_pool[i] = list(myself_item_pool_cols)
        sim_del_histor = {}
        for i in user_similar_jaccard.index:
            sim_del_histor[i] = list(set(sim_users_item_pool[i]) - set(histor_users_item_pool[i]))
        etime = time.time()
        print(f'Obtaining the collection time of items that the user has not interacted with: {etime - stime}s')
        item_idex = []
        user_idex = []
        for i in range(0, len(sim_del_histor)):
            for item_value in sim_del_histor[i]:
                item_idex.append(item_value)
                user_idex.append(i)
                # break

        print('creat user_idex.pt')
        torch.save(user_idex, self.path + '/user_idex.pt')
        print('creat item_idex.pt')
        torch.save(item_idex, self.path + '/item_idex.pt')
        return user_idex, item_idex

    def normalize(self, adj):
        # normalize
        adj_indices = adj.indices()
        diags = torch.sparse.sum(adj, dim=1).to_dense() + 1e-7
        diags = torch.pow(diags, -1)
        diag_lookup = diags[adj_indices[0, :]]

        adj_value = adj.values()
        normal_value = torch.mul(adj_value, diag_lookup)
        normal_adj = torch.sparse.FloatTensor(adj_indices, normal_value,
                                              self.adj_shape).to(self.device).coalesce()

        return normal_adj

    def ssl_triple_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        norm_emb1 = F.normalize(z1)
        norm_emb2 = F.normalize(z2)
        pos_score = torch.mul(norm_emb1, norm_emb2).sum(dim=1)
        ttl_score = torch.matmul(norm_emb1, norm_emb2.transpose(0, 1))
        pos_score = torch.exp(pos_score / self.tau)
        ttl_score = torch.exp(ttl_score / self.tau).sum(dim=1)

        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss

    def cal_cos_sim(self, u_idx, i_idx, eps=1e-8, CHUNK_SIZE=CHUNK_SIZE_FOR_SPMM):
        user_feature = self.get_all_user_embedding().to(self.device)
        item_feature = self.get_all_item_embedding().to(self.device)
        a_n, b_n = user_feature.norm(dim=1)[:, None], item_feature.norm(dim=1)[:, None]
        user_feature_norm = user_feature / torch.max(a_n, eps * torch.ones_like(a_n))
        item_feature_norm = item_feature / torch.max(b_n, eps * torch.ones_like(b_n))

        L = u_idx.shape[0]
        sims = torch.zeros(L, dtype=user_feature.dtype).to(self.device)
        for idx in range(0, L, CHUNK_SIZE):
            a_batch = torch.index_select(user_feature_norm, 0, u_idx[idx:idx + CHUNK_SIZE].type(torch.long))
            b_batch = torch.index_select(item_feature_norm, 0, i_idx[idx:idx + CHUNK_SIZE].type(torch.long))
            dot_prods = torch.mul(a_batch, b_batch).sum(1)
            sims[idx:idx + CHUNK_SIZE] = dot_prods
        return sims

    def get_aug_adj(self):
        sim_mat = self.get_sim_mat()
        sim_value = torch.div(torch.add(sim_mat.values(), 1), 2)
        sim_adj = torch.sparse.FloatTensor(sim_mat.indices(), sim_value, sim_mat.shape).coalesce()
        aug_user, aug_item = self.user_idex_tensor, self.item_idex_tensor
        aug_user = torch.tensor(np.array(aug_user)).to(self.device)
        aug_item = torch.tensor(np.array(aug_item)).to(self.device)

        cos_sim = self.cal_cos_sim(aug_user, aug_item)
        cos_sim_value = torch.div(torch.add(cos_sim, 1), 2)
        cos_sim_indices = torch.stack([aug_user, aug_item], dim=0)
        cos_sim_adj = torch.sparse.FloatTensor(cos_sim_indices.type(torch.long), cos_sim_value,
                                               sim_mat.shape).coalesce()

        cos_sim_adj_dense = cos_sim_adj.to_dense()
        user_topk_add = torch.topk(cos_sim_adj_dense, self.topk_add, dim=-1, largest=True, sorted=True)
        user_topk_add_values = torch.reshape(user_topk_add.values, [-1])
        user_topk_add_columns = torch.reshape(user_topk_add.indices, [-1])
        user_id = []
        for i in range(0, sim_mat.shape[0]):
            for idx in range(0, self.topk_add):
                user_id.append(i)
                # break
        user_id_add = torch.tensor(np.array(user_id)).to(self.device)
        aug_indices = torch.stack([user_id_add, user_topk_add_columns], dim=0)
        sub_aug = torch.sparse.FloatTensor(aug_indices, user_topk_add_values, sim_mat.shape).coalesce()

        aug_ui_inter = (sub_aug + sim_adj).coalesce()

        aug_ui_inter_indices = aug_ui_inter.indices()
        aug_ui_inter_indices2 = torch.stack([aug_ui_inter_indices[0], aug_ui_inter_indices[1] + aug_ui_inter.shape[0]],
                                            dim=0)
        aug_adj_ui_inter = torch.sparse.FloatTensor(aug_ui_inter_indices2, aug_ui_inter.values(),
                                                    self.adj_shape).coalesce()
        aug_adj = (aug_adj_ui_inter + aug_adj_ui_inter.t()).coalesce()
        normal_aug_adj = self.normalize(aug_adj)

        return normal_aug_adj, aug_ui_inter

    def DeNoise(self, pruning=0.0, denoise_adj=None):
        pruned_adj_index = denoise_adj.indices()
        pruned_adj_value = denoise_adj.values()

        denoise_adj_deal = torch.sparse.FloatTensor(pruned_adj_index, pruned_adj_value - 1,
                                                    denoise_adj.shape).coalesce()
        denoise_adj_dense = denoise_adj_deal.to_dense()
        user_topk_del = torch.topk(denoise_adj_dense, self.topk_del, dim=-1, largest=False, sorted=True)
        user_topk_del_columns = torch.reshape(user_topk_del.indices, [-1])
        user_idx = []
        for i in range(0, denoise_adj.shape[0]):
            for idx in range(0, self.topk_del):
                user_idx.append(i)
        user_topk_del_rows = torch.tensor(np.array(user_idx)).to(self.device)
        denoise_indices = torch.stack([user_topk_del_rows, user_topk_del_columns], dim=0)

        pre_value = torch.ones_like(user_topk_del_rows) * 3
        pre_adj_1 = torch.sparse.FloatTensor(denoise_indices, pre_value, denoise_adj.shape).coalesce()
        pre_adj_2 = (pre_adj_1 + denoise_adj).coalesce()
        pre_del_value = pre_adj_2.values()
        pre_del_value_1 = torch.where(pre_del_value >= 3, torch.zeros_like(pre_del_value),
                                      pre_del_value) if 3 > 0 else pre_del_value
        denoise_adj_sparse = torch.sparse.FloatTensor(pre_adj_2.indices(), pre_del_value_1,
                                                      denoise_adj.shape).coalesce()


        denoise_indices_fin = torch.stack(
            [denoise_adj_sparse.indices()[0], denoise_adj_sparse.indices()[1] + denoise_adj.shape[0]], dim=0)
        pruned_adj = torch.sparse.FloatTensor(denoise_indices_fin, denoise_adj_sparse.values() ,
                                              self.adj_shape).coalesce()
        pruned_adj_fin = (pruned_adj + pruned_adj.t()).coalesce()
        pruned_adj_normal = self.normalize(pruned_adj_fin)

        return pruned_adj_normal

    # Train
    def forward(self, epoch_idx=0):
        user_embeddings = self.get_all_user_embedding()
        item_embeddings = self.get_all_item_embedding()
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        embeddings_list = [all_embeddings]

        self.aug_adj, self.denoise_adj_pre = self.get_aug_adj()
        for _ in range(self.n_layers):
            all_embeddings = self.special_spmm(self.aug_adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def ssl_forward(self, pruning=0.0, denoise_adj=None):
        user_embeddings = self.get_all_user_embedding()
        item_embeddings = self.get_all_item_embedding()
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        embeddings_list = [all_embeddings]

        self.Denoise_adj = self.DeNoise(pruning=pruning, denoise_adj=denoise_adj)
        for _ in range(self.n_layers):
            all_embeddings = self.special_spmm(self.Denoise_adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction, epoch_idx, tensorboard):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        # obtain embedding
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward(epoch_idx=epoch_idx)
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        loss = mf_loss

        # calculate L2 reg
        if self.reg_weight > 0.:
            user_embeddings = self.get_all_user_embedding()
            item_embeddings = self.get_all_item_embedding()
            u_ego_embeddings = user_embeddings[user]
            pos_ego_embeddings = item_embeddings[pos_item]
            neg_ego_embeddings = item_embeddings[neg_item]
            reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings).squeeze()
            loss += self.reg_weight * reg_loss

        # calculate agreement
        if self.MIM_weight > 0.:
            denoise_user_all_embeddings, _ = self.ssl_forward(pruning=self.prune_threshold,denoise_adj=self.denoise_adj_pre.detach())
            denoise_u_embeddings = denoise_user_all_embeddings[user]
            mutual_info = self.ssl_triple_loss(u_embeddings, denoise_u_embeddings)
            loss += self.MIM_weight * mutual_info

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        self.restore_user_e, self.restore_item_e = self.forward()

        u_embeddings = self.restore_user_e[user]
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores.view(-1)
