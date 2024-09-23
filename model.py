import numpy as np

import typing
from typing import Any, Tuple

import einops
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tensorflow as tf
import tensorflow_text as tf_text
from layers import *

#This is the main model

class Translator(tf.keras.Model):
    @classmethod 
    def add_method(cls, fun): #this class method allows us to add additional methods to Translator class dynamically we can extend the functionality of the class without modifying its original code
        setattr(cls, fun.__name__, fun)
        return fun
    

    def __init__(self, units, context_text_processor, target_text_processor): #context_text_processor is an instance of TextVectorization used for processing the input text(context), same for target texxt
        super().__init__()

        encoder = Encoder(context_text_processor, units)#instance of encoder class
        decoder = Decoder(target_text_processor, units)#instance of decoder class

        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        context, x = inputs #context is the processed input sequences, x the input sequences for the decoder(e.g. the target language sentences shofted by one step)

        context = self.encoder(context)#the input text is passed throught the encoder, producing a context representation that encapsulates the information from the input sequence
        logits - self.decoder(context, x)#The context output from the encoder and the target input sequence are passed to the decoder, which outputs logits
    
        #handling keras masking

        try:
            del logits._keras_mask #keras may attach a mask to the logots during processing to handle padding in sequences. The attempt to delete this mask is to prevent keras from scaling the loss and accuracy calculations, which could lead to incorrect results
        except AttributeError: #if the mask doesnt exist 
            pass
        

        return logits
    

#test
model = Translator(UNITS, context_text_processor, target_text_processor)

logits = model((ex_context_tok, ex_tar_in))

print(f'Context tokens, shape: (batch, s, units) {ex_context_tok.shape}')
print(f'Target tokens, shape: (batch, t) {ex_tar_in.shape}')
print(f'logits, shape: (batch, t, target_vocabulary_size) {logits.shape}')



#Train 

def masked_loss(y_true, y_pred):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none') #this is the typical loss function for multi-class classification problems where the true labels are integers It calculates the nefative log-likelihood of the true class based on the predicted class probabilities , from_logits = True tells the function that the y_pred input is raw logits(the ouput of the decoder) and needs to be passed through a softmax internally to convert it into probabilities, masking is used to ignore padding tokenss
    loss = loss_fn(y_true, y_pred)

    mask = tf.cast(y_true != 0, loss.dtype)#all non zeros elements in y_true are lept and zero elements are ignored
    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask) #normalization tf.reduce_sum(maks) to ensure that the loss is averaged over valid tokens, not the entire sequence

#custom Masked Accuracy Function
def masked_acc(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis = -1) #The accuracy is the proportion of correctly predicted tokens. To do this, the predicted logits (y_pred) are converted into token indices using tf.argmax(y_pred, axis=-1), which selects the index of the maximum logit for each position (the predicted token).
    y_pred = tf.cast(y_pred, y_true.dtype) 

    match = tf.cast(y_true == y_pred, tf.float32) #After casting the predictions to the same type as y_true, match = tf.cast(y_true == y_pred, tf.float32) checks if the predicted token matches the true token at each position. This returns a tensor of 1s (correct) and 0s (incorrect).
    mask = tf.cast(y_true != 0, tf.float32)#The mask (mask = tf.cast(y_true != 0, tf.float32)) ensures that accuracy is only calculated over non-padding tokens.

    return tf.reduce_sum(match) / tf.reduce_sum(mask) #The final accuracy is the total number of correct predictions divided by the total number of non-padding tokens.

#model.compile configures the model for training by specifying the optimizer, loss function, and metrics to track.
model.compile(optimizer = "adam", #Adam (Adaptive Moment Estimation) is a popular gradient-based optimization algorithm that adapts learning rates for each parameter. It combines the advantages of both SGD with momentum and RMSProp. It's commonly used in deep learning because of its efficiency and good performance.
             loss = masked_loss, #The custom masked_loss is used to ensure that padding tokens do not contribute to the loss during training.
             metrics = [masked_acc, masked_loss])#Similarly, masked_acc and masked_loss are tracked as metrics to monitor both accuracy and the loss, giving an idea of how well the model is learning.

#model Evaluation

vocab_size = 1.0 * target_text_processor.vocabulary_size() #Before training, the model is randomly initialized. The predicted logits (i.e., output distributions) will be close to uniform, meaning the model is not yet "aware" of the correct answers.

{"expected_loss": tf.math.log(vocab_size).numpy, "expected_acc": 1 / vocab_size} #The expected accuracy is 1/vocab_size because with uniform random guessing, the probability of picking the correct word is the inverse of the vocabulary size. Similarly, the expected loss is the log of the vocabulary size, representing the uncertainty of a uniform distribution.


model.evaluate(val_ds, steps=20, return_dict=True)#Evaluation is performed to check how well the model is performing on the validation dataset (val_ds), with the model outputting the loss and accuracy metrics over a specified number of steps.


#Training Process

history = model.fit(
    train_ds.repeat(), #The training process involves feeding the model batches of input-output pairs (train_ds), calculating the loss, and updating the model weights using backpropagation.
    epochs=100, #The model is trained over multiple epochs (complete passes through the dataset). For each epoch, the model sees 100 batches of data (steps_per_epoch=100).
    steps_per_epoch=100,
    validation_data=val_ds,#Validation is performed periodically using val_ds to check how the model generalizes to unseen data, with metrics recorded every 20 steps (validation_steps=20).
    validation_steps=20,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)]#A callback like EarlyStopping is often used to prevent overfitting by stopping training when the model's performance on the validation set no longer improves (after 3 epochs of no improvement).
    )


#plotting the Training Curves
# Training Curves:
# These plots show how the loss and accuracy evolve over the course of training, for both the training and validation datasets.
# Loss (Cross Entropy/token) is plotted against epochs, showing whether the model is minimizing the error on both the training and validation datasets.
# Accuracy plots show how well the model is learning to predict the correct target tokens over time.

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch #')
plt.ylabel('CE/token')
plt.legend()

plt.plot(history.history['masked_acc'], label='accuracy')
plt.plot(history.history['val_masked_acc'], label='val_accuracy')
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch #')
plt.ylabel('CE/token')
plt.legend()
