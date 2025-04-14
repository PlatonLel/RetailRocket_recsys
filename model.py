import torch
import torch.nn as nn

class BPRMatrixFactorization(nn.Module):
    def __init__(self,
                 num_users=None,
                 num_items=None,
                 num_categories=None,
                 num_prop_types=None,
                 num_prop_values=None,
                 embedding_dim=64,
                 prop_embedding_dim=32,
                 dropout_rate=0.1):
        super(BPRMatrixFactorization, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.category_embeddings = nn.Embedding(num_categories, prop_embedding_dim, padding_idx=0)
        self.prop_type_embeddings = nn.Embedding(num_prop_types, prop_embedding_dim, padding_idx=0)
        self.prop_value_embeddings = nn.Embedding(num_prop_values, prop_embedding_dim, padding_idx=0)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim + 3*prop_embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.user_embeddings.weight)
        nn.init.xavier_normal_(self.item_embeddings.weight)
        nn.init.xavier_normal_(self.category_embeddings.weight)
        nn.init.xavier_normal_(self.prop_type_embeddings.weight)
        nn.init.xavier_normal_(self.prop_value_embeddings.weight)

    def _get_item_embeddings(self, item_ids, category_ids, prop_type_ids, prop_value_ids):
        item_emb = self.item_embeddings(item_ids)
        category_emb = self.category_embeddings(category_ids)
        prop_type_emb = self.prop_type_embeddings(prop_type_ids)
        prop_value_emb = self.prop_value_embeddings(prop_value_ids)
        combined_emb = torch.cat([item_emb, category_emb, prop_type_emb, prop_value_emb], dim=1)
        return self.projection(combined_emb)

    def forward(self,
            user_ids,
            pos_item_ids,
            neg_item_ids=None,
            pos_cat=None,
            neg_cat=None,
            pos_prop_type=None,
            pos_prop_value=None,
            neg_prop_type=None,
            neg_prop_value=None):
        user_emb = self.user_embeddings(user_ids)
        pos_item_emb = self._get_item_embeddings(pos_item_ids, pos_cat, pos_prop_type, pos_prop_value)
        pos_score = (user_emb * pos_item_emb).sum(dim=1)
        if neg_item_ids is not None and neg_cat is not None and neg_prop_type is not None and neg_prop_value is not None:
            neg_item_emb = self._get_item_embeddings(neg_item_ids, neg_cat, neg_prop_type, neg_prop_value)
            neg_score = (user_emb * neg_item_emb).sum(dim=1)
            return pos_score - neg_score
        else:
            return pos_score
        
        
