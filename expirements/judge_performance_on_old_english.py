from classifier import judge
from utils.load_datasets import load_old_english_dataset
import matplotlib.pyplot as plt

sj = judge.Noise_Shakespeare_Classifier()
sj.train("noise_dataset/noise.txt", show_report=True)

ds = load_old_english_dataset(0,0)
probas = sj.shakespeare_likeliness(ds)

plt.hist(probas)
plt.xlabel('Probability of Shakespeare')
plt.ylabel('Frequency')
plt.title("Shakespeare likeliness on Old English Dataset")

plt.show()
