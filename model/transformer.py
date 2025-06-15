import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_seq_len):
        super().__init__()
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        pe = torch.zeros((self.max_seq_len, self.model_dim))
        positions = torch.arange(max_seq_len).unsqueeze(1)
        i = torch.arange(0, self.model_dim, 2)
        div_term = 1 / torch.pow(10000, 2 * i  / self.model_dim)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer('positional_encoding', pe)
    def forward(self, tokens):
        seq_len = tokens.shape[1]
        return self.positional_encoding[:seq_len].unsqueeze(0)

class Attention(nn.Module):
    def __init__(self, m_dim, h_dim, apply_mask=False): 
        super().__init__()
        self.m_dim = m_dim
        self.h_dim = h_dim
        self.Wq = nn.Linear(m_dim, h_dim)
        self.Wk = nn.Linear(m_dim, h_dim)
        self.Wv = nn.Linear(m_dim, h_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.apply_mask = apply_mask

    def forward(self, q, k, v, attention_mask=None):
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        out = torch.matmul(q, k.transpose(-2, -1))  / math.sqrt(self.h_dim)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).bool()
            out.masked_fill_(torch.logical_not(mask), -1e9)
        if self.apply_mask:
            seq_len_q, seq_len_k = out.shape[1], out.shape[2]
            mask = torch.tril(torch.ones((seq_len_q, seq_len_k))).bool().to(out.device)
            out.masked_fill_(torch.logical_not(mask.unsqueeze(0)), -1e9)  # unsqueeze
        out = F.softmax(out, dim=-1)
        out = torch.matmul(out, v)
        out = self.dropout(out)
        return out
    

class Encoder(nn.Module):
    def __init__(self, m_dim, n_heads):
        super().__init__()
        self.m_dim = m_dim
        self.n_heads = n_heads
        self.h_dim = m_dim // n_heads
        self.attention_heads = nn.ModuleList([Attention(self.m_dim, self.h_dim) for _ in range(self.n_heads)])
        self.Wo = nn.Linear(self.n_heads * self.h_dim, self.m_dim)
        self.layernorm1 = nn.LayerNorm(self.m_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.m_dim, self.m_dim * 2),
            nn.ReLU(), 
            nn.Linear(self.m_dim * 2, self.m_dim)
        )
        self.layernorm2 = nn.LayerNorm(self.m_dim)
        self.dropout = nn.Dropout(p=0.1)



    def forward(self, input, attention_mask=None):
        out = torch.cat(
            [attention_head(input, input, input, attention_mask) for attention_head in self.attention_heads], 
            dim=-1
        )
        out = self.Wo(out)
        out = self.layernorm1(  out + input)
        out = self.dropout(out)
        fout = self.ffn(out)
        out = self.layernorm2(out + fout)
        out = self.dropout(out)
        return out



class Decoder(nn.Module):
    def __init__(self, m_dim, n_heads):
        super().__init__()
        self.m_dim = m_dim
        self.n_heads = n_heads
        self.h_dim = self.m_dim // self.n_heads
        self.attention_heads = nn.ModuleList([Attention(self.m_dim, self.h_dim, apply_mask=True) for _ in range(self.n_heads)])
        self.Wo = nn.Linear(self.h_dim * self.n_heads, self.m_dim)
        self.layernorm1 = nn.LayerNorm(self.m_dim)
        self.attention_heads2 = nn.ModuleList([Attention(self.m_dim, self.h_dim, apply_mask=False) for _ in range(self.n_heads)])
        self.Wo2 = nn.Linear(self.h_dim * self.n_heads, self.m_dim)
        self.layernorm2 = nn.LayerNorm(self.m_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.m_dim, self.m_dim * 2), 
            nn.ReLU(),
            nn.Linear(self.m_dim * 2, self.m_dim)
        )
        self.layernorm3 = nn.LayerNorm(self.m_dim)
        self.dropout = nn.Dropout(p=0.1)


    
    def forward(self, input, encoder_out, encoder_attention_mask=None, decoder_attention_mask=None):
        out = torch.cat(
            [attention_head(input, input, input, decoder_attention_mask) for attention_head in self.attention_heads], 
            dim=-1
        )
        out = self.Wo(out)

        out1 = self.layernorm1(out + input)
        out1 = self.dropout(out1)

        out = torch.cat(
            [attention_head(out1, encoder_out, encoder_out, encoder_attention_mask) for attention_head in self.attention_heads2],
            dim=-1
        )
        out = self.Wo2(out)

        out1 = self.layernorm2(out1 + out)
        out1 = self.dropout(out1)
        
        out = self.ffn(out1)
        out = self.dropout(out)
        out = self.layernorm3(out + out1)
        out = self.dropout(out)
        return out
    


class Transformer(nn.Module):
    def __init__(self, n_layers, m_dim, n_heads, vocab_size, max_seq_len, device):
        super().__init__()
        self.n_layers = n_layers
        self.m_dim = m_dim
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(self.vocab_size, self.m_dim)
        self.positional_encoding = PositionalEncoding(self.m_dim, self.max_seq_len)
        self.encoder_stack = nn.ModuleList((
            Encoder(self.m_dim, self.n_heads)
            for _ in range(self.n_layers)))
        self.decoder_stack = nn.ModuleList((
            Decoder(self.m_dim, self.n_heads)
            for _ in range(self.n_layers)))
        self.device = device

    def forward(self, encoder_input, decoder_input):
        source_tokens, encoder_attention_mask = encoder_input['source_tokens'].to(self.device), encoder_input['encoder_attention_mask'].to(self.device)
        target_tokens, decoder_attention_mask = decoder_input['target_tokens'].to(self.device), decoder_input['decoder_attention_mask'].to(self.device)
        source_embedding = self.embedding(source_tokens) * math.sqrt(self.m_dim)
        target_embedding = self.embedding(target_tokens) * math.sqrt(self.m_dim)
        source_embedding = source_embedding + self.positional_encoding(source_embedding)
        target_embedding = target_embedding + self.positional_encoding(target_embedding)
        encoder_out = source_embedding
        decoder_output = target_embedding
        for encoder_layer in self.encoder_stack:
            encoder_out = encoder_layer(encoder_out, encoder_attention_mask)
        for decoder_layer in self.decoder_stack:
            decoder_output = decoder_layer(decoder_output, encoder_out, encoder_attention_mask, decoder_attention_mask)
        logits = decoder_output @ self.embedding.weight.T
        return logits
    
    def generate(self, source_tokens, tokenizer, max_len=None):
        if max_len is None:
            max_len = self.max_seq_len

        batch_size = source_tokens.size(0)
        device = source_tokens.device

        # Encode
        source_embedding = self.embedding(source_tokens) * math.sqrt(self.m_dim)
        source_embedding = source_embedding + self.positional_encoding(source_embedding)
        encoder_out = source_embedding
        for encoder_layer in self.encoder_stack:
            encoder_out = encoder_layer(encoder_out)

        # Start with <bos>
        generated_tokens = torch.full(
            (batch_size, 1),
            tokenizer.bos_token_id,
            dtype=torch.long,
            device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            generated_embedding = self.embedding(generated_tokens) * math.sqrt(self.m_dim)
            generated_embedding = generated_embedding + self.positional_encoding(generated_embedding)

            decoder_output = generated_embedding
            for decoder_layer in self.decoder_stack:
                decoder_output = decoder_layer(decoder_output, encoder_out)

            logits = decoder_output @ self.embedding.weight.T
            logits_new_token = logits[:, -1, :]  # (batch, vocab)
            next_token = torch.argmax(logits_new_token, dim=-1).unsqueeze(1)  # (batch, 1)
            if False:
                print(generated_tokens.shape)
                print(generated_tokens)
                print(logits)
                print(logits.argmax(-1))
                print(next_token)
                input()
                
            # If already finished, keep appending <pad>
            next_token[finished] = tokenizer.pad_token_id

            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

            # Update finished mask
            finished = finished | (next_token.squeeze(1) == tokenizer.eos_token_id)
            if finished.all():
                break

        return generated_tokens
    