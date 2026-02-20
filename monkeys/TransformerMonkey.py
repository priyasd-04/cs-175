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
    def __init__(self, vocab_size, tokenizer, n_embd=64, n_head=4, n_layer=3, block_size=128, dropout = 0.2):
        """
        :param vocab_size: character amount
        :param n_embd: Embedding size of each vocabulary word, affects ability to capture more dimensions.
        :param n_head: How many attention heads running in parallel, affects ability to recognize more patterns.
        :param n_layer: How many transformer layers in the model, affects complexity of the model.
        :param block_size: Context window sizes, affects how contextually coherent each output is.
        """
        super().__init__()
        self.block_size = block_size
        self.dropout = nn.Dropout(dropout)
        self.tokenizer = tokenizer
        #Token Embedding Table: The lookup table that each character is mapped to. 
        #Size(vocab_size,n_embed) as in each row is a vocab word with n_embed columns if information
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        #Position Embedding Table: The positional information of each word in each position. 
        #Size(block_size, n_embed): Each position in the context window gets n_embed columns of information.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        #Initialize n_layer count of transformer layers.  
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=n_embd*4, batch_first=True, dropout=dropout)
        
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
        x = self.dropout(tok_emb + pos_emb)

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
            if self.tokenizer.stoi[EOS_TOKEN] == idx_next:
                break
        return idx
    
    @torch.no_grad()
    def estimate_loss(self, data, batch_size, eval_iters=100):
        """ Helper to get a clean average loss over multiple batches """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        out = {}
        self.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            ix = torch.randint(len(data) - self.block_size, (batch_size,))
            x = torch.stack([data[i:i+self.block_size] for i in ix]).to(device)
            y = torch.stack([data[i+1:i+self.block_size+1] for i in ix]).to(device)
            logits, loss = self(x, y)
            losses[k] = loss.item()
        out = losses.mean()
        self.train()
        return out

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



def train_monkey(epochs=5000, batch_size=32, block_size=64, 
                 n_embd=64, n_head=4, n_layer=3, lr=1e-3, dropout=0.2,
                 writer=False, model_path = None, val_split = 0.1, test_split = 0.1):
    """
    Perform full training loop using the TransformerMonkey architecture on the old_english_dataset. 
    Returns (model, tokenizer, test_loss)
    :param epochs: Amount of training cycles
    :param batch_size: (Int) Size of the batches it is trained in
    :param block_size: (Int) The width of the context window
    :param n_embd: (Int) The depth of information given to each embedding
    :param n_head: (Int) The amount of concurrent attention heads
    :param n_layer: (Int) The amount of transformers layered back to back.
    :param lr: (Float) Learning Rate
    :param dropout: (0-1]) Rate that neurons are turned off, performs regularlization well
    :param writer: (Bool) Whether to document growth using Tensorboard
    :param model_path: (Str) The path to save the model to, ignored if writer=False
    :param val_split: (0-1) Portion of data to use for validation
    :param test_split: (0-1) Portion of data to use for testing
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    #load the dataset
    train_ds, val_ds, test_ds = load_old_english_dataset(val_split, test_split, seed=13625442)

    #add EOS_TOKEN to each of the datasets
    all_text = EOS_TOKEN.join(train_ds + val_ds + test_ds) #all text is needed to train the tokenizer
    train_text = EOS_TOKEN.join(train_ds)
    val_text = EOS_TOKEN.join(val_ds)
    test_text = EOS_TOKEN.join(test_ds)

    #turn all of the text data into vectors to encode
    tokenizer = CharTokenizer(all_text)
    train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long).to(device)
    val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long).to(device)
    test_data = torch.tensor(tokenizer.encode(test_text), dtype=torch.long).to(device)

    #initialize model and optimizer with the given parameters
    model = TransformerMonkey(tokenizer.vocab_size, tokenizer=tokenizer, block_size=block_size, n_embd=n_embd,
                               n_head=n_head,n_layer=n_layer,dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    def get_batch(data): #batches are randomized to train in the middle of the dataset, to avoid memorizing starts or ends
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)
    
    # initialize writer to document growth on tensorboard
    if writer:
        try:
            from torch.utils.tensorboard import SummaryWriter  # requires `tensorboard`
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "TensorBoard logging requested (writer=True) but `tensorboard` is not installed. "
                "Install it with: python3 -m pip install tensorboard"
            ) from e
        run_name = f"bs_{block_size}emb{n_embd}_head{n_head}_lyr{n_layer}_lr{lr}_do{dropout}"
        writer = SummaryWriter(f"runs/{run_name}")

    #track least loss to save the best performing model
    min_validation_loss = float('inf')

    #training loop:
    model.train()
    for epoch in range(epochs):
        xb, yb = get_batch(train_data)

        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        #document loss on the writer
        if writer and epoch % 50 == 0:
            training_loss = model.estimate_loss(train_data, batch_size, eval_iters=50)
            validation_loss = model.estimate_loss(val_data, batch_size, eval_iters=50)

            #save the best model if it performs well on validation loss
            if validation_loss < min_validation_loss:
                if model_path:
                    save_dir = Path.cwd() / model_path 
                    save_dir.mkdir(parents=True, exist_ok=True)
                    full_file_path = save_dir / f"{run_name}.pt"
                    model.saveModel(full_file_path)
                min_validation_loss = validation_loss

            writer.add_scalars('Loss', {'train': training_loss,'val': validation_loss,}, epoch)
            
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")


    if writer:
        writer.close()

    #calculate test_loss as a final metric
    test_loss = model.estimate_loss(test_data, batch_size, eval_iters=100)
    return model, tokenizer, test_loss

if __name__ == '__main__':
    print("Loading Datasets...")
    #train_ds = load_old_english_dataset(0, 0, seed=13625442)

    #full_corpus = EOS_TOKEN.join(train_ds) 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training Started...")
    model, tokenizer, test_loss = train_monkey(epochs=1000, batch_size=32, block_size=128)

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