import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
import keras.layers as L
import sys
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers, Sequential, Model

#μετατρέπει το dataset μορφή τετοιά που η πρόβλεψη να μπορεί να βασιστεί σε δεδομένα look_back ημερών
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
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
#διαιρεί το σύνολο δεδομένων σε υποσύνολο train και test
def splitToTrainAnsTestSets(data):
    training_size=int(len(data)*0.65)
    test_size=len(data)-training_size
    train_data,test_data=data[0:training_size,:],data[training_size:len(data),:1]
    return (train_data, test_data)
#Κανονικοποιεί τα δεδομένα
def normalizeDataset(data):
    scaler=MinMaxScaler(feature_range=(0,1))
    data_out = scaler.fit_transform(np.array(data).reshape(-1,1))
    return pd.DataFrame(data_out),scaler

#$python detect.py –d <dataset> -n <number of time series selected>
#Διαβάζει και ελέγχει τις παραμέτρους εισόδου
argc = len(sys.argv)
if argc != 7:
	print("Syntax Error\ndetect.py –d <dataset> -n <number of time series selected> -mae <mae>")
	sys.exit()
#Αρχικοποίηση των παραμέτρων εισόδου
filename = "UnknownFilename"
numberOfSeries = 5
#ορίζεται ότι η πρόβλεψη θα βασίζεται σε 4 παρελθούσες μετρήσεις
look_back = 4
#όριο που καθορίζει τις "εξάρσεις"
mae = 10
for i in range(1,argc):
    if sys.argv[i]=="-d":
        filename = sys.argv[i+1]
    if sys.argv[i]=="-n":
        numberOfSeries = int(sys.argv[i+1])
    if sys.argv[i]=="-mae":
        mae = int(sys.argv[i+1])
if filename == "Unknown File" or numberOfSeries == 0 :
    print("Syntax Error\nforecast.py –d <dataset> -n <number of time series selected>")
    sys.exit()
#Διαβάζει τα δεδομένα από το αρχείο εισόδου
myDataframe = makeDataframe(filename,' ')
#Κρατάμε το επιθυμητο βάθος χρόνου
myDataframe=myDataframe.iloc[:,myDataframe.shape[1]-32:myDataframe.shape[1]-1]
#Βρίσκουμε τη μεγαλύτερη τιμή στο σύνολο δεδομένων
#για να χρησιμοποιηθεί στην κανονικοποίηση
max_element = myDataframe.iloc[0,0]
def absolute_maximum_scale(series):
    return series.abs().max()
for col in myDataframe.columns:
    s = absolute_maximum_scale(myDataframe[col])
    if s > max_element:
        max_element = s
#κανονικοποίηση του συνόλου δεδομένων
myDataframe = myDataframe/max_element
#κρατούνται τα δεδομένα σε επιθυμητό χρόνο (ένα μήνα)
data_series=myDataframe.iloc[:,0:myDataframe.shape[1]-1]
#κρατείται η στήλη - στόχος (ημέρα για πρόβλεψη)
labels = myDataframe.iloc[:,myDataframe.shape[1]-1]
#διάκριση σε train και test σύνολο
train, valid, Y_train, Y_valid = train_test_split(data_series.values, labels.values, test_size=0.35, random_state=0)
X_train = train.reshape((train.shape[0], train.shape[1], 1))
X_valid = valid.reshape((valid.shape[0], valid.shape[1], 1))
#οι διαστάσεις του συνόλου
serie_size = X_train.shape[1]
n_features = X_train.shape[2]
#Παράμετροι της δημιουργίας του μοντέλου
layers = 1 #(x2)
units = 24
epochs = 100
batch = 20
#
encoder_decoder = Sequential()
encoder_decoder.add(L.LSTM(serie_size, activation='relu', input_shape=(serie_size, n_features), return_sequences=True))
for i in range(0,layers):
    encoder_decoder.add(L.LSTM(units, activation='relu', return_sequences=True))
encoder_decoder.add(L.LSTM(1, activation='relu'))
encoder_decoder.add(L.RepeatVector(serie_size))
encoder_decoder.add(L.LSTM(serie_size, activation='relu', return_sequences=True))
for i in range(0,layers):
    encoder_decoder.add(L.LSTM(units, activation='relu', return_sequences=True))
encoder_decoder.add(L.Dropout(0.1/units))
encoder_decoder.add(L.Dense(1))

encoder_decoder.summary()

adam = optimizers.Adam()

encoder_decoder.compile(loss='mse', optimizer=adam)


encoder_decoder_history = encoder_decoder.fit(X_train, X_train, 
                                              batch_size=batch, 
                                              epochs=epochs, 
                                              verbose=2)


for i in range(0,numberOfSeries):
    y = encoder_decoder.predict(X_valid[i].reshape(1,30,1))


    y[0]*max_element
    X_valid[i]*max_element
    
    ABS = y[0]*max_element - X_valid[i]*max_element
    
    ABSTICK = []
    counter = 0
    for e in ABS:
        if abs(e[0]) > mae:
            ABSTICK.append(X_valid[i][counter][0]*max_element)
        else:
            ABSTICK.append(np.nan)
        counter = counter  + 1
    
    
    plt.plot(y[0]*max_element)
    plt.plot(X_valid[i]*max_element)
    plt.plot(ABSTICK,'o',c="black")
    plt.show()





