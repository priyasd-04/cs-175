import sklearn
from utils.load_datasets import load_shakespeare_dataset
from monkeys.generator import random_monkey
from random import randint
from os import makedirs, path
import numpy as np

#just leaving this here, i don't think we'll have a big enough corpus to train a very large
#tokenizer, it'll probably be easy to just use something pre-trained, gpt2 seems nice

# tokenizer idea (not used in baseline):
# tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
# tokens = tokenizer("We should use the gpt2 tokenizer")
# print(tokenizer.decode(tokens["input_ids"]))

class Noise_Shakespeare_Classifier:
    def __init__(self, seed=13625442):
        self.vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(analyzer='char', ngram_range=(2,5))
        self.seed = seed

    def generate_noise(file_path, n=2211, min_length = 30, max_length = 1000): #min and max length taken from dataset, where true min is 34 and true max is 1494
        makedirs(path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            for i in range(n):
                k = randint(min_length, max_length)
                f.write(random_monkey(k) + "\n")

    def train(self, filepath ,show_report = False):
        sp_train, sp_test = load_shakespeare_dataset(0.0, 0.2)

        with open(filepath) as f:
            noise_texts = [line.strip() for line in f.readlines()]
        split_idx = int(len(noise_texts) * 0.8)

        noise_train = noise_texts[:split_idx]
        noise_test = noise_texts[split_idx:]

        train_texts = sp_train + noise_train
        train_labels = np.concatenate([np.ones(len(sp_train)), np.zeros(len(noise_train))])

        test_texts = sp_test + noise_test
        test_labels = np.concatenate([np.ones(len(sp_test)), np.zeros(len(noise_test))])

        print("fitting vectorizer")
        X_train = self.vectorizer.fit_transform(train_texts)
        X_test = self.vectorizer.transform(test_texts)

        print("training logistic regression")
        self.model = sklearn.linear_model.LogisticRegression(max_iter=1000, random_state = self.seed)
        self.model.fit(X_train, train_labels)

        if show_report:
            predictions = self.model.predict(X_test)
            print("\n--- Judge Evaluation Report ---")
            print(sklearn.metrics.classification_report(test_labels, predictions, target_names=["Noise", "Shakespeare"]))

    def shakespeare_likeliness(self, text_input:str):
        if isinstance(text_input, str):
            text_input = [text_input]
        vector_input = self.vectorizer.transform(text_input)

        probabilities = self.model.predict_proba(vector_input) # returns (probability of monkey, probability of shakespeare)

        return [prob[1] for prob in probabilities]


    #def scrappy_judge():
    #    model = sklearn.lo

#scrappy_judge.generate_noise("noise_dataset/noise.txt")
#sj = Noise_Shakespeare_Classifier()
#sj.train("noise_dataset/noise.txt", show_report=True)