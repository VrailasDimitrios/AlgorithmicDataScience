# LSTM για την πρόβλεψη τιμών μετοχών
import sys
import functools
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import math
from sklearn.metrics import mean_squared_error

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
def splitToTrainAndTestSets(data):
    training_size=int(len(data)*0.65)
    test_size=len(data)-training_size
    train_data,test_data=data[0:training_size,:],data[training_size:len(data),:1]
    return (train_data, test_data)
#Κανονικοποιεί τα δεδομένα
def normalizeDataset(data):
    scaler=MinMaxScaler(feature_range=(0,1))
    data_out = scaler.fit_transform(np.array(data).reshape(-1,1))
    return data_out
    


#$python forecast.py –d <dataset> -n <number of time series selected>

#Διαβάζει και ελέγχει τις παραμέτρους εισόδου
argc = len(sys.argv)
if argc != 5:
	print("Syntax Error\nforecast.py –d <dataset> -n <number of time series selected>")
	sys.exit()
#Αρχικοποίηση των παραμέτρων εισόδου
filename = "dataset2.txt"
numberOfSeries = 5
for i in range(1,argc):
    if sys.argv[i]=="-d":
        filename = sys.argv[i+1]
    if sys.argv[i]=="-n":
        numberOfSeries = int(sys.argv[i+1])
if filename == "Unknown File" or numberOfSeries == 0 :
    print("Syntax Error\nforecast.py –d <dataset> -n <number of time series selected>")
    sys.exit()
#ορίζεται ότι η πρόβλεψη θα βασίζεται σε 4 παρελθούσες μετρήσεις
look_back = 4

#Διαβάζει τα δεδομένα από το αρχείο εισόδου
myDataframe = makeDataframe(filename,' ')
#Παράμετοι κατασκευή του μοντέλου πρόβλεψης
layer_size = 50 #Μέγεθος του επιπέδου του νευρωνικού δικτύου
epochs = 100 #Εποχές εκπαίδευσης
batch_size = 3 #Μέγεθος του batch

#Δημιουργία του μοντέλου
#Αποτελείται από τρία LSTM επίπεδα
model=Sequential()
model.add(LSTM(layer_size,return_sequences=True,input_shape=(look_back,1)))
model.add(LSTM(layer_size,return_sequences=True))
model.add(LSTM(layer_size))
model.add(Dropout(1/layer_size))
#η έξοδος του μοντέλου
model.add(Dense(1))
#compilation του μοντέλου
model.compile(loss='mean_squared_error',optimizer='adam')
#θα εξεταστεί η πρόβλεψη για κάθε μία από τις χρονοσειρές εισόδου
for i in range(0,numberOfSeries):
    #μετατρέπει το dataframe σε dataset
    dataset=makeDataset(myDataframe.iloc[i])
    #κανονικοποίηση των τιμών δεδομένων
    df1=normalizeDataset(dataset.astype('float32'))
    #διάκριση σε test και train set
    (trainDataset, testDataset) = splitToTrainAndTestSets(df1)
    #προσαραμογή του βάθους ημερών
    trainX, trainY = create_dataset(trainDataset, look_back)
    testX, testY = create_dataset(testDataset, look_back)
    scaler=MinMaxScaler(feature_range=(0,1))
    #οι διαστάσεις της εισόδου μετασχηματίζονται σε [αρχιθμός χρονοσειρών, βάθος χρόπνου ελέγχου, παράμετροι]
    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
    testX = testX.reshape(testX.shape[0], testX.shape[1], 1)
    #εκπαίδευση του μοντέλου
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0)
    #δημιουργία του scaler για την αποκατάσταση των τιμών από την κανονικοποίηση τους
    scaler=MinMaxScaler(feature_range=(0,1))
    data_out = scaler.fit_transform(np.array(dataset.astype('float32')).reshape(-1,1))
    # make predictions
    testPredict = model.predict(testX)
    # invert predictions
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY.reshape(-1, 1))
    #εμφανίζει τον mean squared error
    print(math.sqrt(mean_squared_error(testY,testPredict)))
    #εμφανίζει το διάγραμμα της παραγμτικής τιμής και αυτής που προβλεύθηκε
    plt.plot(testY)
    plt.plot(testPredict)
    plt.title("Diagram "+str(i+1))
    plt.show()


    