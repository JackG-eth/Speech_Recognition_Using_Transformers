import wget
from tqdm import tqdm
import tarfile
import os

if os.path.isfile("speech_commands_v0.01.tar.gz"):
    print("Dataset already exists.")
else:
    print("Downloading dataset...")
    url = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
    wget.download(url, "speech_commands_v0.01.tar.gz")

print("Starting extraction...")
dataset = tarfile.open(name="speech_commands_v0.01.tar.gz", mode='r:gz')

for member in tqdm(dataset.getmembers()):
    dataset.extract(member, path="dataset/")