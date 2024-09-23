import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
from preprocessing import *
from shapeChecker import *
import matplotlib.pyplot as plt

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
    
    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)#convert inpput texts to tensors
        if len(texts.shape) == 0: #check if texts has zero dimensions
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.text_processor(texts).to_tensor()
        context = self(context)#this line invokes the call method of the encoder processing the tokent Ids through the embedding and GRU layers resul is a tensor with contextual information about input sentence

        return context
    

#testing
# Encode the input sequence.
encoder = Encoder(context_text_processor, UNITS)
ex_context = encoder(ex_context_tok)

print(f'Context tokens, shape (batch, s): {ex_context_tok.shape}')
print(f'Encoder output, shape (batch, s, units): {ex_context.shape}')



class CrossAttention(tf.keras.layers.Layer): #iherits fomr tf.keras.layers.Layer
    def __init__(self, units, **kwargs): #constructor method that initialize the layer units is the dimensionality of the layer **kwargs variable input arguments (variable list of arguments)
        super().__init__() #calling the constructor of the parent class
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs) 
        self.layernorm = tf.keras.layers.LayerNormalization() #normalization layer normalizess the features
        self.add = tf.keras.layers.Add() #creates addition lyer which is used to add the residual connection

    def call(self, x, context): #defines the forward pass of the layer it takes two input x and context the value and key from the encoder
        shape_checker = ShapeChecker() #ensures that the shapes of the inputs the expected format. verifies the the dimensions of tensors at different steps

        shape_checker(x, 'batch t units')#checks the shape of the x tensor t is time dimesion - sewuence length of the query
        shape_checker(context, 'batch s units')#s is the sequence length of the context
        
        #the mha layer computes attention by comparing the query x and the context value=context
        attn_output, attn_scores = self.mha(#attn_output is the result of the attention mechanism its a wighted sum of the context vectors, attn_scores are the attention weights that indicate how much attention wach query element gives to the context element
            query = x,
            value = context,
            return_attention_scores = True#ensures that the atteetion scores are returned for future use (eg. for visualization or debugging)
        )

        shape_checker(x, 'batch t units')
        shape_checker(attn_scores, 'batch heads t s')

        #cache the attention scores for plotting layer
        attn_scores = tf.reduce_mean(attn_scores, axis=1) #averaging attention scores
        shape_checker(attn_scores, 'batch t s')
        self.last_attention_weights = attn_scores

        x = self.add([x, attn_output]) #This is the residual connection. The input x is added to the attn_output, which allows the model to retain information from the original query x while adding in the new information from the attention mechanism.
        x = self.layernorm(x) #After the addition, layer normalization is applied to x to stabilize the training process by ensuring the output has a mean of 0 and variance of 1. This prevents exploding or vanishing gradients and improves convergence.

        return x



####test#####
attention_layer = CrossAttention(UNITS)

# Attend to the encoded tokens
embed = tf.keras.layers.Embedding(target_text_processor.vocabulary_size(),output_dim=UNITS, mask_zero=True)
ex_tar_embed = embed(ex_tar_in)

result = attention_layer(ex_tar_embed, ex_context)

print(f'Context sequence, shape (batch, s, units): {ex_context.shape}')
print(f'Target sequence, shape (batch, t, units): {ex_tar_embed.shape}')
print(f'Attention result, shape (batch, t, units): {result.shape}')
print(f'Attention weights, shape (batch, t, s):    {attention_layer.last_attention_weights.shape}')


#The attention weights will sum to 1 over the context sequence, at each location in the target sequence.
attention_layer.last_attention_weights[0].numpy().sum(axis=-1)
#attention weights across the context sequences at t = 0


attention_weights = attention_layer.last_attention_weights
mask=(ex_context_tok != 0).numpy()

plt.subplot(1, 2, 1)
plt.pcolormesh(mask*attention_weights[:, 0, :])
plt.title('Attention weights')

plt.subplot(1, 2, 2)
plt.pcolormesh(mask)
plt.title('Mask');

plt.show()


#The Decoder
#The decoders's  is to generate predictions for the next token at each location in the target sequence.
#at each time step by attending to the encoded input from the encoder
#1. for each token in the target sequence(in english) the decoder looks up embeddings converting the token IDs into continuos vectors.
#2. the decoder uses a unidorectional GRU to process these embeddings in a left to right order
#3. at each time step the RNN output is used as the 'query' to an attention layer. the attention layer helps the decoder focus on different parts of the encoder's output (context) ot predict the next word
#4. Using the RNN's output, the decoder predicts the next token in the target sequence by applying a fully connected layer (Dense) to generate a probability distribution over the entire vocabulary.

class Decoder(tf.keras.layers.Layer):
    #class initializer initializes all the layers
    @classmethod
    def add_method(cls, fun): #this is utility decorator, you can add methods dynamically to the decoder class. 
        setattr(cls, fun.__name__, fun)#setattr function assigns the function fun as a method of the class
        return fun


    def __init__(self, text_processor, units):
        super(Decoder, self).__init__()

        self.text_processor = text_processor#responsible for converting raw text to tokens
        
        self.vocab_size = text_processor.vocabulary_size()

        self.word_to_id = tf.keras.layers.StringLookup( #creates a stringLookup layer to map words to their corresponding token IDs
            vocabulary = text_processor.get_vocabulary(),
            mask_token = '',  #any word not in vocabulary 
            oov_token = '[UNK]',#will be alotted [UNK] out-of-vocabulary
            
        )

        self.id_to_word = tf.keras.layers.StringLookup(#created another stringloojup layer but inverts the mapping allowing you to convert token IDs back to words, helps in interpreting out models output to readable form
            vocabulary=text_processor.get_vocabulary(),
            mask_token = '',
            oov_token = "[UNK]",
            invert = True
        )

        self.start_token = self.word_to_id('[START]')
        self.end_token = self.word_to_id('[END]')
        self.units = units


        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True) #initializes an embedding layer that converts token IDs into dense vectors of a fixed size units, mask_zero = true means that the zero token will be ignored in computations because it is often used for padding 


        self.rnn = tf.keras.layers.GRU(units,
                                       return_sequences=True, #retuns hidden state output for each time step, necessary for attention mechanisms
                                       return_state=True, #enabkes the GRU last hidden state 
                                       recurrent_initializer='glorot_uniform')#used for initializing weights 
        

        self.attention = CrossAttention(units)#initializes an instance of the CrossAttention class, which will use the ouput from the RNN as the query when attending to the encoder's output this allows the decoder to focus on different paurs of the encoder's output at each time step

        
        self.output_layer = tf.keras.layers.Dense(self.vocab_size)#takes the putput fomr the previous layer (here GRU) and transforms it into a logits vector that corresponds to the vocabulary size each element in this logits vector represents the raw prediction score for each word in the vocabulary
        

#Training the dataset
@Decoder.add_method
def call(self, context, x, state = None, return_state=False):#method in tensorflow keras that defines how the input passes throught the layer forward pass, context is the output from the encoder which holdds the context (or information) extracted from the input sequence, x is the target sequence input for training this is the ground truth target tokens shifted by one, state is the previous state of the decoder's RNN this is optional and passed in during inference to continue the generation process, return_state if True the RNN's internal state is returned along with the output, this is useful for inference where you want to generate tokens one by one
    shape_checker = ShapeChecker()
    shape_checker(x, 'batch t') #the t is the length of the target sequence
    shape_checker(context, 'batch s units') #here s is the length of input sequence units is the size of encoder's hidden state
    

    #convert the input to the decoder which is token ids into embedding vectors
    x = self.embedding(x)#the firststep is to convert the token IDs into embedding vectors
    shape_checker(x, 'batch t units') 


    #RNN processing 
    x, state = self.rnn(x, initial_state = state) #x is the output for each RNN for every timestep(token) in the sequence, state is the final hidden state of the RNN, which can be passes back during the next call to continue the sequence
    shape_checker(x, 'batch t units')

    #applying attention Attention allows the decoder to selectively focus on different parts of the input sequence for each token it generates, rather than relying on a fixed-length context vector. This helps the model handle long sequences more effectively.
    x = self.attention(x, context) #the attention layer has two input x this is the query or RNN ouput which represenets the current state of the decoder, context is the key value pair which represent the encoder's output
    #the attention mechanism computes the weighted avereage of the encoder's output on how relevant each part of the context is to decoder's current state this gives the decoder access to the entire input sequenc with more emphaisis on relevant pairs
    self.last_attention_weights = self.attention.last_attention_weights#returns attention weights that represent the importance of each imput token (from the encoder) to each ouput token(in the target sequence)
    shape_checker(x, 'batch t units')
    shape_checker(self.last_attention_weights, 'batch t s')

    #Logit predictions (logits are he unnormalized raw output scores of neural network before activation function like softmax or sigmoid is applied to convert them into probabilitites)
    logits = self.output_layer(x)
    shape_checker(logits, 'batch t target_vocab_size')#the target_vocab_size is the number of possible tokens in the target language


    if return_state:
        return logits, state
    
    else:
        return logits
    
    #if return_state is set to true the method returns both the predicted logits and the RNN's final state
    #if false (default) then only logits are returned



#Test The decoder
decoder = Decoder(target_text_processor, UNITS)

#The Decoder predicts the next token in the target sequence, given the context from the encoder and the target input tokens up to that point. During training, it works by taking the correct previous tokens (ex_tar_in) and predicting the next token (logits).
logits = decoder(ex_context, ex_tar_in)

# print(f'encoder output shape: (batch, s, units) {ex_context.shape}')
# print(f'input target tokens shape: (batch, t) {ex_tar_in.shape}')
# print(f'logits shape shape: (batch, target_vocabulary_size) {logits.shape}')


















