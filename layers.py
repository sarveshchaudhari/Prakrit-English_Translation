import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
from preprocessing import *
from shapeChecker import *

#at each time step the encoder ouput is combined with decoders output

UNITS = 256

#Goal of the encoder sequence is to process the context sequence into a sequence of vectors that are useful for the decoder as it attempts to predict the next output for each timestep

class Encoder(tf.keras.layers.Layer):
    def __init__(self, text_processor, units):
        super(Encoder, self).__init__()
        self.text_processor = text_processor #ctext_processor converts raw text into token IDs
        self.vocab_size = text_processor.vocabulary_size() #retireves how many unique tokens generated
        self.units = units #number of units in the hidden layer of the model both for the embedding and RNN layers
        

        #the embedding layer converts tokens to vectors
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True)

        #the RNN layer processes those vetors sequentially
        #in BiRNN both directions are processed independently and their results are combined at each time step
        self.rnn = tf.keras.layers.Bidirectional(
            merge_mode='sum',
            layer = tf.keras.layers.GRU(
                units,#dimensionality of output space the number of hidden units in GRU cell
                return_sequences=True,#This indicated that the GRU must return the hidden state output for each time step in the input sequence, rather than just the final hidden state after processing the whole sequence this is crucial for attention
                recurrent_initializer='glorot_uniform'#also known as xavier uniform initializer helps in initialozaing the weights in a way that maintains the scale of gradients across layers avoids vanishing and exploding gradients
            )
        )
    #call methods details how the input data flows through the layers of the encoder
    def call(self, x):
        shape_checker = ShapeChecker()
        shape_checker(x, 'batch s') #checks wether the data flowing thorugh the encoder adheres to the expected dimensions
        
        x = self.embedding(x) #the input tensor x containing token ids is passed through the embedding layer converts it into a embedding vector with shape bat size, s, units
        shape_checker(x, 'batch s units') #check the shape again
        

        #x is passed through the GRU after embedding 
        x = self.rnn(x) #each vector output to the hiddent state of the GRU at the time step the ouput is GRUs understanding of the context
        shape_checker(x, 'batch s units')

        return x
    

#testing
# Encode the input sequence.
encoder = Encoder(context_text_processor, UNITS)
ex_context = encoder(ex_context_tok)

print(f'Context tokens, shape (batch, s): {ex_context_tok.shape}')
print(f'Encoder output, shape (batch, s, units): {ex_context.shape}')





