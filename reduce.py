import csv
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import os, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras import optimizers, Sequential, Model
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets.mnist import load_data
from numpy import reshape
import matplotlib.pyplot as plt

#μετατρέπει το dataset μορφή τετοιά που η πρόβλεψη να μπορεί να βασιστεί σε δεδομένα look_back ημερών
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for j in range(len(dataset)):
        for i in range(len(dataset[j])-look_back-1):
            a = dataset[j][i:(i+look_back)]
            dataX.append(a)
            dataY.append(dataset[j][i + look_back])
    return np.array(dataX), np.array(dataY)
#διαβάζει το αρχείο και δημιουργεί το αντίστοιχο dataframe
def makeDataframe(filename, sep):
    # load the dataset
    dataframe = read_csv(filename, sep=' ', index_col=[0], header=None, engine='python') 
    return dataframe.astype('float32')
#διαβάζει το dataframe και επιστρέφει πίνακα τιμών
def makeDataset(dataframe):
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    return dataset

#$python reduce.py –d <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file>
#Διαβάζει και ελέγχει τις παραμέτρους εισόδου
argc = len(sys.argv)
if argc != 9:
	print("Syntax Error\nreduce.py –d <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file>")
	sys.exit()
#Αρχικοποίηση των παραμέτρων εισόδου
filename = "Unknown File"
queries = "Unknown File"
outdataset = "Unknown File"
outquery = "Unknown File"

numberOfSeries = 5
look_back = 32
mae = 10

timeDepth = 64

for i in range(1,argc):
    if sys.argv[i]=="-d":
        filename = sys.argv[i+1]
    if sys.argv[i]=="-q":
        queries = sys.argv[i+1]
    if sys.argv[i]=="-od":
        outdataset = sys.argv[i+1]
    if sys.argv[i]=="-oq":
        outquery = sys.argv[i+1]
if filename == "Unknown File" or queries == 0 :
    print("Syntax Error\nreduce.py –d <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file>")
    sys.exit()
numberOfSeries = 5
look_back = 32
mae = 10

timeDepth = 67


#Διαάζει τα δεδομένα από τα αρχεία
myDataframe = makeDataframe(filename,' ')
Labels = myDataframe.index.tolist()
myDataframe_query = makeDataframe(queries,' ')
LabelsQ = myDataframe_query.index.tolist()

#Δημιουργεί τα αντίστοιχα dataframe
myDataframe=myDataframe.iloc[:,myDataframe.shape[1]-timeDepth:myDataframe.shape[1]-1]
myDataframe_query=myDataframe_query.iloc[:,myDataframe_query.shape[1]-timeDepth:myDataframe_query.shape[1]-1]
#Κανονικοποίηση των δεδομένων
max_element = myDataframe.iloc[0,0]
max_element_query = myDataframe_query.iloc[0,0]
def absolute_maximum_scale(series):
    return series.abs().max()
for col in myDataframe.columns:
    s = absolute_maximum_scale(myDataframe[col])
    if s > max_element:
        max_element = s
myDataframe = myDataframe/max_element

for col in myDataframe_query.columns:
    s = absolute_maximum_scale(myDataframe_query[col])
    if s > max_element:
        max_element = s
myDataframe_query = myDataframe_query/max_element_query


#Κρατάμε τα τελευτάια timeDepth δείγματα
myDataframe=myDataframe.iloc[:,0:myDataframe.shape[1]-1]
myDataframe_query=myDataframe_query.iloc[:,0:myDataframe_query.shape[1]-1]
#διαμορφώνουμε το "σχήμα" των δεδομένων
myDataframe = myDataframe.values.reshape(-1,timeDepth-2)
myDataframe_query = myDataframe_query.values.reshape(-1,timeDepth-2)

#Υπεισέρχεται η παράμετρος του look back
datasetX, datasetY = create_dataset(myDataframe, look_back)
datasetX_query, datasetY_query = create_dataset(myDataframe_query, look_back) 


datasetX, datasetY = create_dataset(myDataframe, look_back)
datasetX_query, datasetY_query = create_dataset(myDataframe_query, look_back)  
#διαμορφώνεται το σύνολο των δεδομένων σε 4 διαστάσεις
datasetX=datasetX.reshape(myDataframe.shape[0],datasetX.shape[1],-1,1)
datasetX_query=datasetX_query.reshape(myDataframe_query.shape[0],datasetX_query.shape[1],-1,1)

#διάκριση σε test και train set
train, test, trainY, testY = train_test_split(datasetX, Labels, test_size=0.25, random_state=0)

validQuery = datasetX_query


serie_size =  train.shape[0]
n_features =  train.shape[1]
#παράμετροι μοντέλου
layers = 2
units = 24
#επιθυμητή διάσταση
latent_dim = 8

#tf.compat.v1.disable_eager_execution()
inputshape = Input(shape=(train.shape[1], train.shape[2],1))

#κατασκευή του μοντέλου
enc_conv1 = Conv2D(units, (2, 2), activation='relu', padding='same')(inputshape)          
enc_pool1 = MaxPooling2D((2, 2), padding='same')(enc_conv1)
enc_dropout1 = Dropout(1/units)(enc_pool1)

for i in range(0,layers):
    enc_dropout2 = Dropout(1/units)(enc_pool1)
    if i == 0:
        enc_conv2 = Conv2D(units, (2, 2), activation='relu', padding='same')(enc_dropout1)
    else:
        enc_conv2 = Conv2D(units, (2, 2), activation='relu', padding='same')(enc_dropout2)
    enc_pool2 = MaxPooling2D((4, 4), padding='same')(enc_conv2)
    enc_dropout2 = Dropout(1/units)(enc_pool2)
    enc_conv3 = Conv2D(units, (2, 2), activation='relu', padding='same')(enc_dropout2)
    enc_pool3 = MaxPooling2D((4, 4), padding='same')(enc_conv3)
    enc_dropout3 = Dropout(1/units)(enc_pool3)
    
enc_pre = Dense(latent_dim)(enc_dropout3)
#έξοδος του κωδικοποιητή
enc_out = Flatten()(enc_pre)

dec_conv0 = Conv2D(units, (4, 4), activation='relu', padding='same')(enc_pre)
dec_upsample0 = UpSampling2D((4, 4))(dec_conv0)
dec_dropout0 = Dropout(1/units)(dec_upsample0)
dec_conv1 = Conv2D(units, (4, 4), activation='relu', padding='same')(dec_dropout0)
dec_upsample1 = UpSampling2D((4, 4))(dec_conv1)
dec_dropout1 = Dropout(1/units)(dec_upsample1)

for i in range(0,layers):
    dec_dropout2 = Dropout(1/units)(dec_upsample1)
    if i==0:
        dec_conv2 = Conv2D(units, (4, 4), activation='relu', padding='same')(dec_dropout1)
    else:
        dec_conv2 = Conv2D(units, (4, 4), activation='relu', padding='same')(dec_dropout2)
    dec_upsample2 = UpSampling2D((2, 2))(dec_conv2)
    dec_dropout2 = Dropout(1/units)(dec_upsample2)
dec_output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(dec_dropout2)

#δημιουργία του κωδικοποιητή
encoder = Model(inputshape, enc_out)
encoder.compile(optimizer='rmsprop', loss='mse') 
#δημιουργία του κωδικοποιητή-αποκωδικοποιητή
autoencoder = Model(inputshape, dec_output)
autoencoder.compile(optimizer='rmsprop', loss='mse') 
#παράμετροι εκπαίδευσης
batch = 16
epochs = 1

autoencoder.fit(train, train, epochs=epochs, batch_size=batch, shuffle=True)
#δημιουργία των συμπιεσμενων δεδομένων
out = encoder.predict(test)*max_element
outq = encoder.predict(validQuery)*max_element_query
#εγγραφή σε αρχεία
#δημιουργία των αρχείων εξόδου
f = open(outdataset+"tmp", 'w')
#δημιουργία
writer = csv.writer(f, delimiter ='\t')
#εγγραφή στο αρχείο
for e in out:
    writer.writerow(e)
f.close()
           
f = open(outquery+"tmp", 'w')
#δημιουργία
writer = csv.writer(f, delimiter ='\t')
#εγγραφή στο αρχείο
for e in outq:
    writer.writerow(e)
f.close()


#input file
fin = open(outquery+"tmp", "rt")
print("open:"+outquery+"tmp")
#output file to write the result to
fout = open(outquery, "wt")
#for each line in the input file
count = 0
for line in fin:
    
	#read replace the string and write to output file
    if (len(line.strip())>0):
        fout.write(LabelsQ[count]+"\t"+line)
        count = count + 1
#close input and output files
fin.close()
fout.close()

#input file
fin = open(outdataset+"tmp", "rt")
#output file to write the result to
fout = open(outdataset, "wt")
#for each line in the input file
count = 0
for line in fin:
	#read replace the string and write to output file
    if (len(line.strip())>0):
        fout.write(Labels[count]+"\t"+line)
        count = count + 1
#close input and output files
fin.close()
fout.close()






