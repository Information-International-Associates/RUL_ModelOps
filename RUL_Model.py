from keras.models import load_model
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *

# Scale Test
def scale(df):
    #return (df - df.mean())/df.std()
    return (df - df.min())/(df.max()-df.min())

# Load Model
model = load_model('ClassifierV2_Smoothing01.h5')
w = [200,175,150,125,100,75,50,40,30,25,20,15,10,5] #Bin definitions associated with current models
        
##Create dictionary of category labels to bin midpoints:
rul_cats = [210]
for i in range(len(w)-1):
    upper , lower = w[i], w[i+1]
    rul_cats.append(round(np.mean([upper,lower])))
rul_cats.append(3)
rul_dict = dict(zip(range(len(w)+1), rul_cats))

### Generate Sequences ### 
sequence_length = 50

def gen_sequence(id_df, seq_length, seq_cols):

    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,142),(50,192)
    # 0 50 (start stop) -> from row 0 to row 50
    # 1 51 (start stop) -> from row 1 to row 51
    # 2 52 (start stop) -> from row 2 to row 52
    # ...
    # 141 191 (start stop) -> from row 141 to 191
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

def gen_labels(id_df, seq_length, label):
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    # I have to remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target.
    return data_matrix[seq_length:num_elements, :]


def rec_plot(s, eps=0.10, steps=10):
    d = pdist(s[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return Z

def pad_series(test_df, sequence_length=sequence_length):
    groupby = test_df.groupby('id')['cycle'].max()
    nrows = groupby[groupby<=50]
    over_50 = test_df[~test_df.id.isin(nrows.index)]

    for unit in nrows.index:
        temp = test_df[test_df.id == unit]
        padding = pd.DataFrame()
        nmissing = 51 - len(temp)
        #Create synthetic starter rows
        for i in range(nmissing):
            padding = padding.append(pd.DataFrame(temp.iloc[0]).transpose(), ignore_index=True)
        #Combine synthetic starter padding with available data
        temp = padding.append(temp, ignore_index=True)
        #Renumber cycles
        temp.cycle = range(1,len(temp)+1)
        #Append new padded series to over 50, 
        over_50 = over_50.append(temp, ignore_index=True)
    #Reorder dataframe by id and cycle
    over_50 = over_50.sort_values(by=['id', 'cycle'])
    return over_50

def preprocess(test_df):
    ### Rename and Parse raw columns
    if len(test_df.columns) > 2:
        test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
    test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                         's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                         's15', 's16', 's17', 's18', 's19', 's20', 's21']
    ### SCALE TEST DATA ###
    for col in test_df.columns:
        if col[0] == 's':
            test_df[col] = scale(test_df[col])
    test_df = test_df.dropna(axis=1)

    ### Pad Sequences with under 51 cycles
    test_df = pad_series(test_df)

    ### GENERATE X TEST ###
    x_test = []
    #Currently Hard Coded based on features showing variance in train file 1
    sequence_cols = ['setting1', 'setting2', 's2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    for engine_id in test_df.id.unique():

        for sequence in gen_sequence(test_df[test_df.id==engine_id], sequence_length, sequence_cols):
            x_test.append(sequence)
        
    x_test = np.asarray(x_test)

    ### TRANSFORM X TRAIN TEST IN IMAGES ###

    x_test_img = np.apply_along_axis(rec_plot, 1, x_test).astype('float16')

    return test_df, x_test_img

# Generate Class Predictions
def predict(x_test_img):
    "Returns the class labels, one for each sliding window."
    return model.predict_classes(x_test_img)

def preprocess_and_predict(test_df, output='RUL', sequence_length=sequence_length):
    "Returns one prediction per unit. Output can be 'RUL' or 'Category'."
    test_df, x_test_img = preprocess(test_df)
    class_predictions = predict(x_test_img)
    test_df = test_df[test_df['cycle']>sequence_length]
    test_df['class_prediction'] = class_predictions
    test_df['RUL_prediction'] = test_df['class_prediction'].map(rul_dict)
    return test_df

def summarize_predictions_by_unit(test_df_output, col='RUL_prediction'):
    "Input should already have class_prediction and RUL_prediction columns. You can chain this with preprocess_and_predict. For example: rul_predictions = summarize_predictions_by_unit(preprocess_and_predict(test_df))"
    return test_df_output.groupby('id').tail(1)[col].values

def test(test_df):
    rul_predictions = summarize_predictions_by_unit(preprocess_and_predict(test_df))
    return rul_predictions

def my_test():
    print('testing')
#Retraining

def preprocess_train(train_df, w = [5]):
    if len(train_df.columns) == 28:
        train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
    train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                         's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                         's15', 's16', 's17', 's18', 's19', 's20', 's21']
    print('#id:',len(train_df.id.unique()))
    train_df = train_df.sort_values(['id','cycle'])

    max_cycle = train_df['cycle'].max()
    ### CALCULATE RUL TRAIN ###
    train_df['RUL']=train_df.groupby(['id'])['cycle'].transform(max)-train_df['cycle']

    train_df['label'] = np.where(train_df['RUL'] <= w[0], 1, 0 )
    for i in range(1,len(w)):
        train_df.loc[train_df['RUL'] <= w[i], 'label'] = i+1
        
    train_df = train_df[train_df['cycle'] < max_cycle - w[-1]]

    ### SCALE Train DATA ###
    for col in train_df.columns:
        if col[0] == 's':
            train_df[col] = scale(train_df[col])

    train_df = train_df.dropna(axis=1)
    ### Pad Sequences with under 51 cycles
    train_df = pad_series(train_df)
    
    
    ### SEQUENCE COL: COLUMNS TO CONSIDER ###
    # sequence_cols = []
    # for col in train_df.columns:
    #     if col[0] == 's':
    #         sequence_cols.append(col)
            
    # print(sequence_cols)
    #Currently Hard Coded based on features showing variance in train file 1
    sequence_cols = ['setting1', 'setting2', 's2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']

    ### GENERATE X TRAIN ###
    x_train = []
    for engine_id in train_df.id.unique():
        for sequence in gen_sequence(train_df[train_df.id==engine_id], sequence_length, sequence_cols):
            x_train.append(sequence)

    x_train = np.asarray(x_train)

    ### TRANSFORM X TRAIN TEST IN IMAGES ###
    x_train_img = np.apply_along_axis(rec_plot, 1, x_train).astype('float16')


    ### GENERATE Y TRAIN ###
    y_train = []
    for engine_id in train_df.id.unique():
        for label in gen_labels(train_df[train_df.id==engine_id], sequence_length, ['label'] ):
            y_train.append(label)
        
    y_train = np.asarray(y_train).reshape(-1,1)

    ### ENCODE LABEL ###
    y_train = to_categorical(y_train,2)

    return x_train_img, y_train

def train_model(x_train_img, y_train, w = [5], sequence_length = 50, smoothing_coef=0.01):
    n_cats = len(w) + 1

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(sequence_length, sequence_length, 17)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_cats, activation='softmax'))

    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=smoothing_coef)

    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', mode='auto', restore_best_weights=True, verbose=1, patience=6)

    model.fit(x_train_img, y_train, batch_size=512, epochs=25, callbacks=[es], validation_split=0.2, verbose=2)

    return model
