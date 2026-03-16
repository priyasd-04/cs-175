# Author(s): Priya Deshmukh
# Random monkey generation, baseline monkey
import random
import string

ALPHABET = string.ascii_lowercase + " .,;:!?"

# Generate a random string of fixed length.
def random_monkey(length = 25):
    return ''.join(random.choice(ALPHABET) for _ in range(length))