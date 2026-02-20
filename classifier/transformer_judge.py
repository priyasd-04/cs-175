import torch
import torch.nn as nn
from monkeys.transformer_monkey import TransformerMonkey, CharTokenizer
from utils.load_datasets import load_old_english_dataset, load_shakespeare_dataset


class TransformerJudge(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.transformer = base_model
        # freeze base model weights to prevent them from being updated during training
        for param in self.transformer.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(base_model.lm_head.in_features, 128), #input layer
            nn.ReLU(), # activation function
            nn.Dropout(0.3), # drop out to prevent overfitting
            nn.Linear(128, 1) # output, gives likelihood of being Shakespeare
        )
        
    def generate_noise(self, monkey_model, tokenizer, device, n_samples=500):
        noise_texts = []
        monkey_model.eval()
        with torch.no_grad():
            for _ in range(n_samples):
                seed = torch.zeros((1, 1), dtype=torch.long, device=device) # start with single token
                generated = monkey_model.generate(seed, max_new_tokens=200) # generate up to 200 tokens
                text = tokenizer.decode(generated[0].tolist())
                noise_texts.append(text)
                
        return noise_texts
    def forward(self, idx):
        # use transformer to get token embeddings
        with torch.no_grad():
            token_embeddings = self.transformer.token_embedding_table(idx)
            pos_embeddings = self.transformer.position_embedding_table(torch.arange(idx.shape[1], device=idx.device))
            x = token_embeddings + pos_embeddings
            
            # create mask for prior inputs
            mask = nn.Transformer.generate_square_subsequent_mask(idx.shape[1]).to(idx.device)
            x = self.transformer.blocks(x, mask=mask, is_causal=True)
            x = self.transformer.ln_f(x)
            
        # Take final token representation for classification
        last_hidden = x[:, -1, :] 
        
        # pass thru classifier to get logits
        logits = self.classifier(last_hidden)
        
        # sigmoid function to convert logits --> probability
        return torch.sigmoid(logits) 
    
    
def train_transformer_judge(judge_model, tokenizer, device, epochs=5):
    sp_train, sp_test = load_shakespeare_dataset(0.0, 0.2)
    
    noise_texts = judge_model.generate_noise(monkey_model=TransformerMonkey(tokenizer.vocab_size).to(device), tokenizer=tokenizer, device=device, n_samples=500)

    texts = sp_train + sp_test + noise_texts # combine shakespeare + noise for training
    
    labels = torch.cat([
        torch.ones(len(sp_train)),
        torch.zeros(len(noise_texts))
    ])
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(judge_model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    
    for epoch in range(epochs):
        for text, label in zip(texts, labels):
            encode = torch.tensor(tokenizer.encode(text), dtype=torch.long).unsqueeze(0).to(device)
            output = judge_model(encode)
            loss = loss_fn(output.squeeze(), label.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()