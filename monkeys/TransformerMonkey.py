import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from utils.load_datasets import load_old_english_dataset

from pathlib import Path
import pickle

EOS_TOKEN = '\t'

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.chars.append(EOS_TOKEN)

        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }

    def encode(self, s):
        return [self.stoi[c] for c in s]
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])
    
    def saveTokenizer(self, path):
        """Saves the tokenizer, expecting path of structure Path.cwd() / folder / tokenizerName.pkl"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def loadTokenizer(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    

class TransformerMonkey(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=3, block_size=128):
        """
        :param vocab_size: character amount
        :param n_embd: Embedding size of each vocabulary word, affects ability to capture more dimensions.
        :param n_head: How many attention heads running in parallel, affects ability to recognize more patterns.
        :param n_layer: How many transformer layers in the model, affects complexity of the model.
        :param block_size: Context window sizes, affects how contextually coherent each output is.
        """
        super().__init__()
        self.block_size = block_size
        #Token Embedding Table: The lookup table that each character is mapped to. 
        #Size(vocab_size,n_embed) as in each row is a vocab word with n_embed columns if information
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        #Position Embedding Table: The positional information of each word in each position. 
        #Size(block_size, n_embed): Each position in the context window gets n_embed columns of information.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        #Initialize n_layer count of transformer layers.  
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, batch_first=True)
        
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        #LayerNorm keeps our vectors normalized (~std deviation of 1)
        self.ln_f = nn.LayerNorm(n_embd)

        #Linear maps the embedings back to vocab words
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape #Batch, Time -> The amount of sentences, the width of the context window

        #Translate the input into embedded vectors
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        #create a mask for prior inputs
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)

        #run the transformer layers
        x = self.blocks(x, mask, is_causal=True)

        #run the normalization layer
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None: #if we're training, calculate cross_entropy
            B, T, C = logits.shape #Batch, Time, embed_length
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, stop_token = EOS_TOKEN, temperature = 0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond) #run the model on everything in context

            logits = logits[:, -1, :] #pluck the last logit
            probs = F.softmax(logits/temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # choose a random idx to generate
            idx = torch.cat((idx, idx_next), dim=1) # add the idx to the data
            #if tokenizer.stoi[EOS_TOKEN] == idx_next:
                #break
        return idx
    
    def saveModel(self, path):
        """Save the model, expecting a path of structure path.cwd() / folder / modelName.pt"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def loadModel(self, path):
        path = Path(path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(torch.load(path, weights_only=True, map_location=device))
        self.to(device)


def train_monkey(text_data, epochs=5000, batch_size=32, block_size=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    tokenizer = CharTokenizer(text_data)
    data = torch.tensor(tokenizer.encode(text_data), dtype=torch.long)

    model = TransformerMonkey(tokenizer.vocab_size, block_size=block_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    def get_batch(): #batches are randomized to train in the middle of the dataset, to avoid memorizing starts or ends
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)
    
    model.train()
    for epoch in range(epochs):
        xb, yb = get_batch()

        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    return model, tokenizer, device

if __name__ == '__main__':
    print("Loading Datasets...")
    train_ds = load_old_english_dataset(0, 0, seed=13625442)

    full_corpus = EOS_TOKEN.join(train_ds) 

    print(f"Training Started...")
    model, tokenizer, device = train_monkey(full_corpus, epochs=1000, batch_size=32, block_size=128)

    model.eval() # Switch to evaluation mode (turns off dropout)
    
    
    prompts = ["To be, or not to be", "The", "Romeo, where for out thou Romeo", "My familys blacksmith ", "Today I "]
    with torch.no_grad():
        for prompt in prompts:
            print(f"\nPrompt: '{prompt}'")
        
            context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
            
                
            generated_indices = model.generate(context, max_new_tokens=150)
            output_text = tokenizer.decode(generated_indices[0].tolist())
            print(f"Generated: {output_text}")

    print("Saving Monkey...")
    path = Path.cwd() / "models" / "SmallTransformerMonkey.pt"
    model.saveModel(path)

    print("Saving Tokenizer...")
    path = Path.cwd() / "tokenizers" / "oldEnglishCharTokenizer.pkl"
    tokenizer.saveTokenizer(path)

    print(f"Save Successful! {path}")