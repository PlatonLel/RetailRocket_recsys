import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class BPRMatrixFactorization(nn.Module):
    def __init__(self,
                 num_users=None,
                 num_items=None,
                 embedding_dim=64,
                 dropout_rate=0.1):
        super(BPRMatrixFactorization, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.user_embeddings.weight)
        nn.init.xavier_normal_(self.item_embeddings.weight)

    def _get_item_embeddings(self, item_ids):
        item_emb = self.item_embeddings(item_ids)
        return self.dropout(item_emb)

    def forward(self, user_ids, pos_item_ids, neg_item_ids=None, *args, **kwargs):
        user_emb = self.user_embeddings(user_ids)
        pos_item_emb = self._get_item_embeddings(pos_item_ids)
        pos_score = (user_emb * pos_item_emb).sum(dim=1)
        if neg_item_ids is not None:
            neg_item_emb = self._get_item_embeddings(neg_item_ids)
            neg_score = (user_emb * neg_item_emb).sum(dim=1)
            return pos_score - neg_score
        else:
            return pos_score
        
