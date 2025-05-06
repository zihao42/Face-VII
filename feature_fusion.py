import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossModalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        return self.norm(attn_output + query)


class ModalFusionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, n_modality, dropout=0.1):
        super(ModalFusionBlock, self).__init__()
        self.n_modality = n_modality
        self.fusion_blocks = nn.ParameterList([
            nn.ParameterList(CrossModalAttention(embed_dim, num_heads) for _ in range(self.n_modality))
            for _ in range(self.n_modality)])
        self.linear1 = nn.ParameterList([nn.Linear(embed_dim * self.n_modality, embed_dim)
                                         for _ in range(self.n_modality)])
        self.linear2 = nn.ParameterList([nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        ) for _ in range(self.n_modality)])
        self.norm = nn.ParameterList([nn.LayerNorm(embed_dim)
                                      for _ in range(self.n_modality)])
        self.dropout = nn.ParameterList([nn.Dropout(dropout)
                                         for _ in range(self.n_modality)])

    def forward(self, modal_data):
        assert len(modal_data) == self.n_modality, \
            f"Mismatch of number of modality with data. " \
            f"Predefined number is {self.n_modality}, but get {len(modal_data)}"

        result = []
        for i in range(self.n_modality):
            aligned = [None] * self.n_modality
            for j in range(self.n_modality):
                aligned[j] = (self.fusion_blocks[i][j](modal_data[i], modal_data[j], modal_data[j]))

            concat = torch.cat(aligned, dim=-1)
            x = self.linear1[i](concat)
            x = self.norm[i](x + self.dropout[i](self.linear2[i](x)))
            result.append(x)

        return result


class MultimodalTransformer(nn.Module):
    def __init__(self, modality_num, num_classes, feature_only=False, input_dim=768,
                 embed_dim=128, num_heads=8, num_layers=1):

        super(MultimodalTransformer, self).__init__()
        self.feature_only = feature_only
        self.n_modality = modality_num
        self.linear = nn.ParameterList(
            [nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU()
            ) for _ in range(self.n_modality)]
        )
        # add to expose embed_dim --> avoid magic number
        self.embed_dim = embed_dim

        self.layers = nn.ParameterList([ModalFusionBlock(embed_dim, num_heads, n_modality=self.n_modality)
                                        for _ in range(num_layers)])

        if not feature_only:
            # Classification head
            self.classifier = nn.Linear(embed_dim * self.n_modality, num_classes)

    def forward(self, modal_data):
        assert len(modal_data) == self.n_modality, \
            f"Mismatch of number of modality with data. " \
            f"Predefined number is {self.n_modality}, but get {len(modal_data)}"

        for i in range(self.n_modality):
            modal_data[i] = self.linear[i](modal_data[i])

        for layer in self.layers:
            modal_data = layer(modal_data)

        modal_summary = []
        for modal in modal_data:
            modal_summary.append(modal[..., -1, :])

        concat = torch.cat(modal_summary, dim=-1)
        if self.feature_only:
            return concat

        output = self.classifier(concat)
        return output, concat
