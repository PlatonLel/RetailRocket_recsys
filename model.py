import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceModel(nn.Module):
    def __init__(self,
                 num_items=None,
                 num_categories=None,
                 num_prop_types=None,
                 num_prop_values=None,
                 embedding_dim=64,
                 prop_embedding_dim=32,
                 hidden_dim=128,
                 dropout_rate=0.1):
        super(SequenceModel, self).__init__()

        self.item_embeddings = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.category_embeddings = nn.Embedding(num_categories, embedding_dim, padding_idx=0)

        self.prop_type_embeddings = nn.Embedding(num_prop_types, prop_embedding_dim, padding_idx=0)
        self.prop_value_embeddings = nn.Embedding(num_prop_values, prop_embedding_dim, padding_idx=0)

        combined_embedding_dim = (embedding_dim * 2) + (prop_embedding_dim * 2)

        self.embedding_dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(combined_embedding_dim, hidden_dim, batch_first=True)
        self.final_dropout = nn.Dropout(dropout_rate)

        self.category_predictor = nn.Linear(hidden_dim, num_categories)
        item_predictor_in_features = hidden_dim + embedding_dim
        self.item_predictor = nn.Linear(item_predictor_in_features, num_items)
    def forward(self,
            item_sequences=None, 
            category_sequences=None, 
            prop_type_sequences=None, 
            prop_value_sequences=None):


        item_emb = self.item_embeddings(item_sequences)
        category_emb = self.category_embeddings(category_sequences)
        prop_type_emb = self.prop_type_embeddings(prop_type_sequences)
        prop_value_emb = self.prop_value_embeddings(prop_value_sequences)

        combined_emb = torch.cat([item_emb, category_emb, prop_type_emb, prop_value_emb], dim=2)
        combined_emb = self.embedding_dropout(combined_emb)

        _, final_hidden_state = self.gru(combined_emb)

        final_hidden_state = final_hidden_state.squeeze(0)
        final_hidden_state = self.final_dropout(final_hidden_state)

        category_logits = self.category_predictor(final_hidden_state)
        with torch.no_grad():
            predicted_category = torch.argmax(category_logits, dim=1)
        predicted_category_emb = self.category_embeddings(predicted_category)
        context_vector = torch.cat([final_hidden_state, predicted_category_emb], dim=1)
        item_logits = self.item_predictor(context_vector)


        return category_logits, item_logits
