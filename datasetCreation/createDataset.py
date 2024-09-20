import numpy as np

import pathlib

import tensorflow as tf
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


#we will create a tf.data.Dataset of string from the array of strings that is context and target
#this will help us in shufling the data and distribute them in batches effectively

#we will also split our dataset into training and validation dataset


BUFFER_SIZE = len(context_raw)
BATCH_SIZE = 64 #can use 32 too

#we will create a boolean mask to distribute the data into training and validation
#np.random.uniform outputs the random samples from uniform distrbution and return random samples as numpy array
#it will generate random floating point numbers between 0 and 1
#since we are splitting the data to 80% training and 20% validation we are using the <0.8 which means that 80% of the values in the array will be true 


is_train = np.random.uniform(size=(len(target_raw))) < 0.8

#now splitting the dataset into validation and training

train_raw = (tf.data.Dataset
             .from_tensor_slices((context_raw[is_train], target_raw[is_train])) #this create a tf.data.Dataset object from slices of the context_raw and target_raw arrays the is_train is a mask that selects the value from the context and target array if true 
             .shuffle(BUFFER_SIZE) #shuffle the entire dataset
             .batch(BATCH_SIZE)) # Instead of processing each sample individually, the dataset will be divided into batches, allowing the model to process 64 samples at a time.

val_raw = (
    tf.data.Dataset
    .from_tensor_slices((context_raw[~is_train], target_raw[~is_train]))
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE))


for example_context_strings, example_target_strings in train_raw.take(1):
  print(example_context_strings[:5])
  print()
  print(example_target_strings[:5])
  break