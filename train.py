from tqdm import tqdm
from data.wmt14_loader import WMTDataset, collate_fn
from tokenizer.tokenizer import BPETokenizer
from torch.utils.data import DataLoader
from model.transformer import Transformer
from torch.nn import CrossEntropyLoss
import torch
from nltk.translate.bleu_score import corpus_bleu


def train(model, dataloader, loss_fn, optimizer, num_epochs, device):
    model.train()
    step = 1
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} / {num_epochs}")
        for batch in pbar:
            encoder_input, decoder_input = batch['encoder_input'], batch['decoder_input']
            loss_tokens = batch['model_target'].to(device)

            logits = model(encoder_input, decoder_input)
            
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                loss_tokens.view(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            step += 1
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")

def compute_bleu(model, dataloader, tokenizer, device):
    model.eval()
    references = []
    hypotheses = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating BLEU"):
            encoder_input = batch['encoder_input']
            model_target = batch['model_target']
            src_tokens = encoder_input['source_tokens'].to(device)
            outputs = model.generate(src_tokens, tokenizer)
            # outputs: (batch, seq_len)
            for ref, hyp in zip(model_target, outputs):
                # Remove padding
                ref_tokens = [tok for tok in ref.tolist() if tok != tokenizer.pad_token_id]
                hyp_tokens = [tok for tok in hyp.tolist() if tok != tokenizer.pad_token_id]
                # Decode to string, then split to word tokens
                ref_str = tokenizer.decode(ref_tokens, skip_special_tokens=True)[0]
                hyp_str = tokenizer.decode(hyp_tokens, skip_special_tokens=True)[0]
                references.append([ref_str.split()])
                hypotheses.append(hyp_str.split())
    bleu = corpus_bleu(references, hypotheses)
    print(f"BLEU score on training set: {bleu:.4f}")



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Loading tokenizer')
    tokenizer = BPETokenizer.load("tokenizer/WMTTokenizer_v1")
    print('Vocab_length, ', len(tokenizer.vocab))
    print('loading WMT dataset')
    train_dataset = WMTDataset('train[:10_000]')
    print(tokenizer.special_tokens)
    print(
        tokenizer.bos_token_id, 
        tokenizer.eos_token_id,
        tokenizer.unk_token_id,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=64, collate_fn= lambda batch: collate_fn(batch, tokenizer, max_seq_len=128), drop_last=True, num_workers=4)
    print('initalizing transformer model')
    transformer = Transformer(
        n_layers=4,
        m_dim=128,
        n_heads=8,
        vocab_size=len(tokenizer.vocab),
        max_seq_len=512,
        device=device
    ).to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-9)
    loss_fn = CrossEntropyLoss( ignore_index=tokenizer.pad_token_id)
    for p in transformer.parameters():
        if p.dim()>1: torch.nn.init.xavier_uniform_(p)
    optimizer = torch.optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=1e-3)
    print('Model Traning')
    train(transformer, train_dataloader, loss_fn, optimizer, 200, device)
    print('Computing BLUE Score')
    compute_bleu(transformer, train_dataloader, tokenizer, device)

