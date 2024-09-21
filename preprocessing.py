import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
from datasetCreation.createDataset import *

# tf.config.set_visible_devices([], 'GPU') #ignore the Nvidia GPU warning if you dont have gpu

#we are creating a model that can be daved throught tf.saved_model
#it should take tf.string input and tf.string outputs
#we are going to process the text using tf.keras.layers.TextVectorization it maps features to integer sequence
#textVectorization first we tokenize the sentence for eg. "I am Sarvesh" will be tokenized too "I", "am", "Sarvesh"

#Standardization - transforming the text into a consistent format, removing punctuations or special characters

#1. Unicode Normalization 
def tf_lower_and_split_punct(text):
    # Split accented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)
    #we will add a [START] and [END] sequence to the front and the back of the sentence
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

#Now we will do text vectorization
#text Vectorization convert raw texts to their numerical representaion

max_vocab_size = 5000 #this is the maximum vocabulary size maximum number of unique tokens that the vocabulary can copntain


#tf.keras.layers.TextVectorization this is a layer that convert raw texdt to vectors of numbers 

context_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct, #standardized function that we wrote above
    max_tokens=max_vocab_size, #the maximum unique tokens that will be generated
    ragged=True #handles inputs of variable lengths without needing to pad them to fixed sizes
)
#The above is prakrit TexVectorization layer

context_text_processor.adapt(train_raw.map(lambda context, target: context))

# Here are the first 10 words from the vocabulary:


target_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size,
    ragged=True
)

target_text_processor.adapt(train_raw.map(lambda context, target: target))


#the .adapt helps in creating the vocabulary

def process_text(context, target):
    context = context_text_processor(context).to_tensor() #this converts input context strings to token IDs based on the vocabulary learned during the adap() pahse and to_tensor converts the output into padded tensor format
    target = target_text_processor(target) #similar to above except we do not use the to_tensor immediately
    targ_in = target[:,:-1].to_tensor() #tag_in is the sequence to the model we remove the last  token <end> because it doesnt predict anything
    targ_out = target[:,1:].to_tensor() #the targ_out sequence represents the ground truth or the expected output of the model.
    #the to_tensor make sure that the length of these vectors are uniform and is padded with zeroes whereever it is necessary
    return (context, targ_in), targ_out

train_ds = train_raw.map(process_text, tf.data.AUTOTUNE) #map applies process_text function to each element of the train_raw dataset, the process_text function tokenizes each eelement of the dataset, the tf.data.AUTOTUNE allows tensorflow to tune the number of parallel threads used to process the data
val_ds = val_raw.map(process_text, tf.data.AUTOTUNE)


for (ex_context_tok, ex_tar_in), ex_tar_out in train_ds.take(1):
    print(ex_context_tok[0, :10].numpy()) 
    print()
    print(ex_tar_in[0, :10].numpy()) 
    print(ex_tar_out[0, :10].numpy())

  








