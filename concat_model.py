from numpy import load

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *

from spektral.layers import GraphConvSkip, GlobalAvgPool
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.layers.pooling import MinCutPool
from spektral.utils.convolution import normalized_adjacency
from spektral.utils.data import Batch

# Load data
X_train_A, y_train_A = load('tr_ma.npy', allow_pickle=True), load('tr_may.npy', allow_pickle=True)
X_test_A, y_test_A = load('te_ma.npy', allow_pickle=True), load('te_may.npy', allow_pickle=True)
X_val_A, y_val_A = load('val_ma.npy', allow_pickle=True), load('val_may.npy', allow_pickle=True)

X_train_B, A_train_B, y_train_B = load('tr_feat.npy', allow_pickle=True), list(load('tr_adj.npy', allow_pickle=True)), load('tr_class.npy', allow_pickle=True)
X_test_B, A_test_B, y_test_B = load('te_feat.npy', allow_pickle=True), list(load('te_adj.npy', allow_pickle=True)), load('te_class.npy', allow_pickle=True)
X_val_B, A_val_B, y_val_B = load('val_feat.npy', allow_pickle=True), list(load('val_adj.npy', allow_pickle=True)), load('val_class.npy', allow_pickle=True)

# Preprocessing adjacency matrices for convolution
A_train_B = [normalized_adjacency(a) for a in A_train_B]
A_val_B = [normalized_adjacency(a) for a in A_val_B]
A_test_B = [normalized_adjacency(a) for a in A_test_B]

# Load pre-trained models
model_A = load_model('model_A.h5')
model_B = load_model('model_B.h5', custom_objects={'GraphConvSkip': GraphConvSkip, 'MinCutPool': MinCutPool,
                                                'GlobalAvgPool': GlobalAvgPool})

# Get the output before last layer of both models
model_A = Model(inputs=model_A.inputs, outputs=model_A.layers[-2].output)
model_B = Model(inputs=model_B.inputs, outputs=model_B.layers[-2].output)

# Merged two models and add sigmoid layer as last layer
def final_model():
    concat = Concatenate(-1)([model_A.output, model_B.output]) # merge outputs
    concat = Dense(1, activation='sigmoid')(concat)

    model = Model(inputs=[model_A.inputs, model_B.inputs], outputs=concat)
    adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics= ['acc'])
    return model

# Construct the merged model
model = final_model()

# Constructing model_B training input data
X_tr, A_tr, I_tr = Batch(A_train_B, X_train_B).get('XAI')
A_tr = sp_matrix_to_sp_tensor(A_tr)

# Constructing model_B validating input data
X_val, A_val, I_val = Batch(A_val_B, X_val_B).get('XAI')
A_val = sp_matrix_to_sp_tensor(A_val)

# Fit the model
print("Fitting the model")
model.fit([X_train_A, [X_tr, A_tr, I_tr]], y_train_A, validation_data=([X_val_A, [X_val, A_val, I_val]], y_val_A), 
          epochs=2, batch_size=1, verbose=1)

