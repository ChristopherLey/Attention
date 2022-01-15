import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert embed_size % heads == 0, f'Embedding dimension ({embed_size}) ' \
                                        f'should be divisible by nr. of heads ({heads})'

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        """
        TODO: compare performance with https://github.com/pbloem/former/blob/master/former/modules.py
        """
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)
        query = self.queries(query)
        keys = self.keys(keys)

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # quieries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        # Attention(Q,K,V) = softmax(Q*K^T/sqrt(d_k))*V
        attention = torch.softmax(energy/(self.embed_size ** 0.5), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # after einsum (N, query_len, heads, head_dim) then flatten
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)  # Takes average for every example
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))     # Where query is the skip connection in the reference paper
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))         # Where x is the skip connection in the reference paper
        return out


class Encoder(nn.Module):
    def __init__(
            self,
            source_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length

    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(source_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion)
            for _ in range(num_layers)
        ])
        self.dropout = nn.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, sequence_length = x.shape
        positions = torch.arange(0, sequence_length).expand(N, -1).to(self.device)

        input = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            encoded = layer(input, input, input, mask)

        return encoded


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, source_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, source_mask)
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            target_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoded, source_mask, target_mask):
        N, sequence_length = x.shape
        positions = torch.arange(0, sequence_length).expand(N, -1).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.positional_embedding(positions)))

        for layer in self.layers:
            x = layer(x, encoded, encoded, source_mask, target_mask)

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            source_vocab_size,
            target_vocab_size,
            source_pad_index,
            target_pad_index,
            embed_size=512,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0.5,
            device="cuda",
            max_length=100
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            source_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            target_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.source_pad_index = source_pad_index
        self.target_pad_index = target_pad_index
        self.device = device

    def make_source_mask(self, source):
        source_mask = (source != self.source_pad_index).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, source_length)
        return source_mask.to(self.device)

    def make_target_mask(self, target):
        N, target_length = target.shape
        target_mask = torch.tril(torch.ones((target_length, target_length))).expand(
            N, 1, target_length, target_length
        )
        return target_mask.to(self.device)

    def forward(self, source, target):
        source_mask = self.make_source_mask(source)
        target_mask = self.make_target_mask(target)
        encoded_source = self.encoder(source, source_mask)
        out = self.decoder(target, encoded_source, source_mask, target_mask)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    with torch.no_grad():
        out = model(x, trg[:, :-1])
    print(out.shape)
