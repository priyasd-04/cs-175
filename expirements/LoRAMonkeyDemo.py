from monkeys.TransformerMonkey import TransformerMonkey, CharTokenizer
from monkeys.LoRAMonkey import LoRAMonkey
from pathlib import Path
import torch
from utils.load_datasets import load_shakespeare_dataset
from random import randint
from classifier.transformer_judge import TransformerJudge

print("loading models...")
tokenizer = CharTokenizer.loadTokenizer(Path.cwd() / 'tokenizers' / 'oldEnglishCharTokenizer.pkl')
model = TransformerMonkey(tokenizer.vocab_size, tokenizer,block_size=256,
    n_embd=128,
    n_head=16,
    n_layer=12,
    dropout=0.4)


model.loadModel(Path.cwd() / 'models' / 'experiments1' / 'bs_256emb128_head16_lyr12_lr0.001_do0.4.pt')

print(model.tokenizer.vocab_size)

print("loading data...")
train_data, val_data, test_data = load_shakespeare_dataset()
print(f"Average Train Data Length: {sum([len(x) for x in train_data])/len(train_data)}")
print(f"Average Val Data Length: {sum([len(x) for x in val_data])/len(val_data)}")
print(f"Average Test Data Length: {sum([len(x) for x in test_data])/len(test_data)}")
test_data = '\t'.join(test_data)

print("Loading transformer judge...")

ckpt = torch.load('models/transformer_judge.pt', map_location='cuda')

judge_tokenizer = ckpt["tokenizer"]
cfg = ckpt["base_model_config"]

base_for_judge = TransformerMonkey(
    judge_tokenizer.vocab_size, 
    tokenizer=judge_tokenizer,
    block_size=int(cfg["block_size"]), 
    n_embd=int(cfg["n_embd"]),
    n_head=int(cfg["n_head"]), 
    n_layer=int(cfg["n_layer"])
).to('cuda')

judge = TransformerJudge(base_for_judge).to('cuda')
judge.load_state_dict(ckpt["judge_state_dict"])
judge.eval()

results = {}

prompts = []
for i in range(21):
    random = randint(0,len(test_data)-1)
    prompts.append(test_data[random][:25])


for training_proportion in [10,25,50,75,100]:

    slice_idx = int(len(train_data) * (training_proportion / 100))
    train_data_slice = train_data[:slice_idx]

    print(f"training LoRA model... {training_proportion}")
    
    lora = LoRAMonkey(model, rank=32, alpha=32).to('cuda')
    lora.fit(train_data_slice, val_data, lr=5e-5, epochs=3000)

    lora.eval()
    print("Evaluating with judge...")
    total_score = 0
    loss = 0
    
    with torch.no_grad():
        for prompt in prompts:
            context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device='cuda')
            
            #model_generated_indices = model.generate(context, max_new_tokens=150)
            lora_generated_indices = lora.generate(context, max_new_tokens=280)
            lora_output = tokenizer.decode(lora_generated_indices[0].tolist())

            filtered_text = "".join([c for c in lora_output if c in judge_tokenizer.stoi])

            judge_ids = judge_tokenizer.encode(filtered_text)

            if len(judge_ids) > judge.transformer.block_size:
                judge_ids = judge_ids[-judge.transformer.block_size:]
            
            judge_inputs = torch.tensor([judge_ids], dtype=torch.long, device='cuda')
            judge_score = judge(judge_inputs).item()
            total_score += judge_score

            
            loss = lora.estimate_loss(torch.tensor(lora.tokenizer.encode(test_data)).to('cuda'), batch_size=32)
            
            #model_output = tokenizer.decode(model_generated_indices[0].tolist())
            #lora_output = tokenizer.decode(lora_generated_indices[0].tolist())
            #print(f"Old English Monkey Generated: {model_output}")

    total_score = total_score/len(prompts)
    results[training_proportion] = (total_score, loss)
    print(f'{training_proportion}: Loss: {loss} | Score: {total_score}')
    lora.train()
            

print(results)
#total_params = sum(p.numel() for p in lora.parameters())
#trainable_params = sum(p.numel() for p in lora.parameters() if p.requires_grad)
#print(f'Total number of parameters: {total_params}')
#print(f'Total number of trainable parameters: {trainable_params}')

