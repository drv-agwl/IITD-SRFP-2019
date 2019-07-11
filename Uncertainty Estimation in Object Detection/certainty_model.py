import pandas as pd
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, BatchNormalization
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import resample
#from imblearn.over_sampling import SMOTE
import os as os
from tqdm import tqdm


model = Sequential()
model.add(Dense(32, activation = 'relu', kernel_initializer = "glorot_uniform", kernel_regularizer = l2(0.01), input_dim = 10))
model.add(BatchNormalization())
model.add(Dense(64, activation = 'relu', kernel_initializer = "glorot_uniform", kernel_regularizer = l2(0.01)))
model.add(BatchNormalization())
model.add(Dense(128, activation = 'relu', kernel_initializer = "glorot_uniform", kernel_regularizer = l2(0.01)))
model.add(BatchNormalization())
model.add(Dense(256, activation = 'relu', kernel_initializer = "glorot_uniform", kernel_regularizer = l2(0.01)))
model.add(BatchNormalization())
model.add(Dense(128, activation = 'relu', kernel_initializer = "glorot_uniform", kernel_regularizer = l2(0.01)))
model.add(BatchNormalization())
model.add(Dense(64, activation = 'relu', kernel_initializer = "glorot_uniform", kernel_regularizer = l2(0.01)))
model.add(BatchNormalization())
model.add(Dense(32, activation = 'relu', kernel_initializer = "glorot_uniform", kernel_regularizer = l2(0.01)))
model.add(BatchNormalization())
model.add(Dense(1, activation = "sigmoid", kernel_initializer = "glorot_uniform", kernel_regularizer = l2(0.01)))

print(model.summary())

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



chunksize = 10**8
counter = 0
for data in tqdm(pd.read_csv('train_coco.csv', chunksize = chunksize)):

    if counter == 1:
        break

    counter += 1

    data = shuffle(data)

    #DOWNSAMPLING MAJORITY CLASS
    print(data.target.value_counts())
    n_samples = data.target.value_counts()[1]

    data_majority = data[data.target==0]
    data_minority = data[data.target==1]

    data_majority_downsampled = resample(data_majority, 
                                 replace=False,   
                                 n_samples=n_samples,
                                 random_state=123)

    data_downsampled = pd.concat([data_majority_downsampled, data_minority])

    data_downsampled.target.value_counts()

    data = data_downsampled


    X_train = data.iloc[:,2:12]
    Y_train = data.iloc[:,12]

    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.1, shuffle = True)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)


    history = model.fit(X_train, Y_train, validation_data = [X_test, Y_test], batch_size = 128, epochs = 1000, shuffle = True)


    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


#UPSAMPLING MINIORITY CLASS

#data_majority = data[data.target==0]
#data_minority = data[data.target==1]

#data_minority_upsampled = resample(data_minority, 
#                                 replace=True,
#                                 n_samples=39960,
#                                random_state=123)

#data_upsampled = pd.concat([data_majority, data_minority_upsampled])

#data_upsampled.target.value_counts()

#data = data_upsampled






#SMOTE
#sm = SMOTE(random_state=2)
#X_train, Y_train = sm.fit_sample(X_train, Y_train)


#print("After OverSampling, counts of label '1': {}".format(sum(Y_train==1)))
#print("After OverSampling, counts of label '0': {}".format(sum(Y_train==0)))




#PERFORMANCE ON UPSAMPLED TEST DAATASET
#sm = SMOTE(random_state=2)
#X_test2, Y_test2 = sm.fit_sample(X_test, Y_test)

test = model.predict(X_test)

#print("After OverSampling, counts of label '1': {}".format(sum(Y_test==1)))
#print("After OverSampling, counts of label '1': {}".format(sum(test>=0.5)))



## TXT FILE FOR MAP CALCULATION
#test_file = pd.read_csv('test.csv')[:5395].iloc[:,[1, 13, 14, 15, 16]]
test_data = pd.read_csv('val_coco.csv').iloc[:,2:12]
test_data = sc.transform(test_data)

scores = model.predict(test_data, verbose = 1)
test_df = pd.DataFrame(scores, columns = ['predictions'])
test_df.to_csv('predictions.csv')

#test_file = list(np.array(test_file))
#scores = list(np.array(scores))

#for i in tqdm(range(len(test_file))):
#    l = list(test_file[i])
#    if l[0][:-4]+'.txt' in os.listdir('./prediction_results_neural_network'):
#        f = open('./prediction_results_neural_network/'+l[0][:-4]+'.txt', 'a+')
#        f.write('cow ')
#        f.write(str(scores[i][0])+' ')
#        f.write(str(l[1])+' ')
#        f.write(str(l[2])+' ')
#        f.write(str(l[3])+' ')
#        f.write(str(l[4])+'\n')
#        f.close()
#    else:
#        f = open('./prediction_results_neural_network/'+l[0][:-4]+'.txt', 'w')
#        f.write('cow ')
#        f.write(str(scores[i][0])+' ')
#        f.write(str(l[1])+' ')
#        f.write(str(l[2])+' ')
#        f.write(str(l[3])+' ')
#        f.write(str(l[4])+'\n')
#        f.close()
        
#count = 0
#for i in range(len(scores)):
#    if float(scores[i])>=0.5:
#        count+=1
#print(count)
