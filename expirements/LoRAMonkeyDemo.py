from monkeys.TransformerMonkey import TransformerMonkey, CharTokenizer
from monkeys.LoRAMonkey import LoRAMonkey
from pathlib import Path
import torch
from utils.load_datasets import load_shakespeare_dataset

print("loading models...")
tokenizer = CharTokenizer.loadTokenizer(Path.cwd() / 'tokenizers' / 'oldEnglishCharTokenizer.pkl')
model = TransformerMonkey(tokenizer.vocab_size, tokenizer,block_size=256,
    n_embd=128,
    n_head=16,
    n_layer=12,
    dropout=0.4)



model.loadModel(Path.cwd() / 'models' / 'experiments1' / 'bs_256emb128_head16_lyr12_lr0.001_do0.4.pt')

print("loading data...")
train_data, val_data, test_data = load_shakespeare_dataset()

lora = LoRAMonkey(model, rank=16, alpha=32).to('cuda')

print("training LoRA model...")
lora.fit(train_data, val_data, writer=True, model_path='/models/LoRAMonkeys/5000Epochs')

prompts = ["To be, or not to be ", "The", "Romeo, where for out thou Romeo", "All the worlds a stage",
            "Computer Science is ", "E tu Brutus? ", "Uneasy lies the head that wears ", "Ive got that summertime sadness "]
with torch.no_grad():
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device='cuda')
        
            
        model_generated_indices = model.generate(context, max_new_tokens=150)
        lora_generated_indices = lora.generate(context, max_new_tokens=150)

        model_output = tokenizer.decode(model_generated_indices[0].tolist())
        lora_output = tokenizer.decode(lora_generated_indices[0].tolist())
        print(f"Old English Monkey Generated: {model_output}")
        print(f"Lora Monkey Generated: {lora_output}")

    