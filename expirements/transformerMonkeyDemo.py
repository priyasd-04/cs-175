from monkeys.TransformerMonkey import CharTokenizer, TransformerMonkey
from utils.load_datasets import load_old_english_dataset
from pathlib import Path
import torch

tokenizer = CharTokenizer.loadTokenizer(Path.cwd() / 'tokenizers' / 'oldEnglishCharTokenizer.pkl')
model = TransformerMonkey(tokenizer.vocab_size, block_size=128)
model.loadModel(Path.cwd() / 'models' / 'SmallTransformerMonkey.pt')

model.eval() # Switch to evaluation mode (turns off dropout)
    
    
prompts = ["To be, or not to be ", "The", "Romeo, where for out thou Romeo", "All the worlds a stage",
            "Computer Science is ", "E tu Brutus? ", "Uneasy lies the head that wears ", "Ive got that summertime sadness "]
with torch.no_grad():
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device='cuda')
        
            
        generated_indices = model.generate(context, max_new_tokens=150)
        output_text = tokenizer.decode(generated_indices[0].tolist())
        print(f"Generated: {output_text}")