# Authors: Davin Makris
# Experiment shows finetuning different sizes of models, ultimately settling on the final hyperparameters shown in main.
# Took ~12 hours on a RTX3070 gpu to train. 

from monkeys.TransformerMonkey import train_monkey

#Past Models
"""
model, tokenizer, final_test_loss = train_monkey(
    epochs=5000,
    batch_size=16,
    block_size=32,
    n_embd=64,
    n_head=4,
    n_layer=3,
    lr=1e-3,
    dropout=0.3,
    writer=True,
    model_path="models/experiments1",
    val_split=0.1,
    test_split=0.1
)

print(f"Training finished. Final test performance: {final_test_loss:.4f}")

model, tokenizer, final_test_loss = train_monkey(
    epochs=10000,
    batch_size=64,
    block_size=128,
    n_embd=64,
    n_head=4,
    n_layer=4,
    lr=1e-3,
    dropout=0.3,
    writer=True,
    model_path="models/experiments1",
    val_split=0.1,
    test_split=0.1
)

print(f"Training finished. Final test performance: {final_test_loss:.4f}")
model, tokenizer, final_test_loss = train_monkey(
    epochs=20000,
    batch_size=64,
    block_size=128,
    n_embd=128,
    n_head=8,
    n_layer=6,
    lr=1e-3,
    dropout=0.4,
    writer=True,
    model_path="models/experiments1",
    val_split=0.1,
    test_split=0.1
)

print(f"Training finished. Final test performance: {final_test_loss:.4f}")

"""


if __name__ == '__main__':
    model, tokenizer, final_test_loss = train_monkey(
    epochs=50000,
    batch_size=128,
    block_size=256,
    n_embd=128,
    n_head=16,
    n_layer=12,
    lr=1e-3,
    dropout=0.4,
    writer=True,
    model_path="models/experiments1",
    val_split=0.1,
    test_split=0.1
)