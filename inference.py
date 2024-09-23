import numpy as np

import typing
from typing import Any, Tuple

import einops
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tensorflow as tf
import tensorflow_text as tf_text
from layers import *

#inference is the process of feeding a trained neural network with new data and receiving an output. This process is used to make predictions or solve tasks based on the information learned during training.

#getting initial state
#this method initializes the state of the decoder for generating sequences based on the context provided by the encoder
@Decoder.add_method
def get_initial_state(self, context):#context is the output from the encoder which contains information about the input sequence
    batch_size = tf.shape(context)[0] #retrieves the number of sequences in the batch. It determines how many sequences the decoder will process simultaneously
    start_tokens = tf.fill([batch_size, 1], self.start_token)#indicates the beginning of a sequence for the decoder.
    done = tf.zeros([batch_size, 1], dtype=tf.bool)#a tensor initialized to 0 False status indicating the generation of that sequence has finished
    embedded = self.embedding(start_tokens) #converts the start tokens into their corresponding embedding vectors
    return start_tokens, done, self.rnn.get_initial_state(embedded)[0] #return the start_tokens, done status, and the initial state if the RNN, the inintial state is derived from the embedded start tokns allowing the RNN to begin processing


#Tokens to Text

@Decoder.add_method
def tokens_to_text(self, tokens):#this method converts a sequence of token IDs back into a human-readanle string
    words = self.id_to_word(tokens)#converts the token IDs into their corresponding words using a mapping function id_to_word
    result = tf.strings.reduce_join(words, axis=-1, separator = " ")#joins the list of words into a single string with spaces separating them
    result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
    result = tf.strings.regex_replace(result, ' *\[END\] *$', '')#the above two lines remove speacial tokens  
    return result #returns the clean string representation of the generated tokens

@Decoder.add_method
def get_next_token(self, context, next_token, done, state, temperature = 0.0):#This method generates the next token in the sequence based on the current context, previous tokens, and the decoder's state.
    #logit calculation
    logits, state = self( #calls the decoder with the current context and next token to generate logits, which represent the raw prediction score for the next tokent in the target vocabulary
        context, next_token,
        state = state,
        return_state = True
    )

    if temperature == 0.0:  #temperature contains the randomness of the predictions when temperature is 0.0 it uses deterministic approach
        next_token - tf.argmax(logits, axis = -1) #selects the next token with the highest logit score
    
    else:#when temperature is greater than 0.0 it introduces randomnes
        logits = logits[:, -1, :] / temperature #this divides the logits by the temperature, softening the distribution, and samples the next token based on the adjusted probabilitites
        next_token = tf.random.categorical(logits, num_samples = 1)

    done = done | (next_token == self.end_token) #updates the done status for the sequences that have generated an end_token, indicating they have completed their generation
    next_token = tf.where(done, tf.constant(0, dtype = tf.int64), next_token)#if a sequence is marked as done it replaces the next token with 0 , ensuring that the completed sequences do not produce further tokens

    return next_token, done, state #finally the merhod returns the generated next token, the updated done status and the current state of the 



#test
# Setup the loop variables.
next_token, done, state = decoder.get_initial_state(ex_context)
tokens = []

for n in range(10):
  # Run one step.
  next_token, done, state = decoder.get_next_token(
      ex_context, next_token, done, state, temperature=1.0)
  # Add the token to the output.
  tokens.append(next_token)

# Stack all the tokens together.
tokens = tf.concat(tokens, axis=-1) # (batch, t)

# Convert the tokens back to a a string
result = decoder.tokens_to_text(tokens)
print(result[:3].numpy())
