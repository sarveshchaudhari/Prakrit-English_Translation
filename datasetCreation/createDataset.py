import numpy as np

import pathlib

#path to the file is an object to the file dataset.txt which is a tab separated collection of data

path_to_file = pathlib.Path("dataset.txt")

def load_data(path):
    text = path.read_text(encoding = "utf-8")
    lines = text.splitlines()
    triplets = [line.strip().split("\t") for line in lines]
    context = np.array([context.strip("\"") for id, target, context in triplets])
    target = np.array([target.strip("\"") for id, target, context in triplets])

    
    return target, context

target_raw, context_raw = load_data(path_to_file)

print(context_raw[-1])
print("##################")
print(target_raw[-1])