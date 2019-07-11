import pandas as pd
from keras.models import Model
from keras.layers import Activation, Dropout, Dense, BatchNormalization, Input, Conv2D, Flatten, Concatenate, ZeroPadding2D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import os as os
from keras.utils import plot_model
from tqdm import tqdm

x_input = Input(shape = (2, 2, 1024))
x = BatchNormalization()(x_input)
x = ZeroPadding2D(((2, 1), (2, 1)))(x)
x = Conv2D(128, (1, 1), input_shape = (5, 5, 1024))(x)
x = BatchNormalization()(x)
x = Model(inputs = x_input, outputs = x)

x_input_2 = Input(shape = (9,))
x_2 = Model(inputs = x_input_2, outputs = x_input_2)


x_input_3 = Input(shape = (5, 5, 128))
x_3 = BatchNormalization()(x_input_3)
x_3 = Model(inputs = x_input_3, outputs = x_3)

x_4 = Concatenate()([x.output, x_3.output])
x_4 = Flatten()(x_4)
x_4 = Model(inputs = [x_input, x_input_3], outputs = x_4)

combined = Concatenate()([x_4.output, x_2.output])
combined = Dense(32, activation = 'relu', kernel_initializer = "glorot_uniform", kernel_regularizer = l2(0.01), input_dim = 265)(combined)
combined = BatchNormalization()(combined)
combined = Dropout(0.2)(combined)

combined = Dense(64, activation = 'relu', kernel_initializer = "glorot_uniform", kernel_regularizer = l2(0.01), input_dim = 265)(combined)
combined = BatchNormalization()(combined)
combined = Dropout(0.1)(combined)

combined = Dense(128, activation = 'relu', kernel_initializer = "glorot_uniform", kernel_regularizer = l2(0.01), input_dim = 265)(combined)
combined = BatchNormalization()(combined)

combined = Dense(256, activation = 'relu', kernel_initializer = "glorot_uniform", kernel_regularizer = l2(0.01), input_dim = 265)(combined)
combined = BatchNormalization()(combined)

combined = Dense(128, activation = 'relu', kernel_initializer = "glorot_uniform", kernel_regularizer = l2(0.01), input_dim = 265)(combined)
combined = BatchNormalization()(combined)

combined = Dense(64, activation = 'relu', kernel_initializer = "glorot_uniform", kernel_regularizer = l2(0.01), input_dim = 265)(combined)
combined = BatchNormalization()(combined)

combined = Dense(32, activation = 'relu', kernel_initializer = "glorot_uniform", kernel_regularizer = l2(0.01), input_dim = 265)(combined)
combined = BatchNormalization()(combined)

output = Dense(1, activation = "sigmoid", kernel_initializer = "glorot_uniform", kernel_regularizer = l2(0.01))(combined)

model = Model(inputs = [x_input, x_input_2, x_input_3], outputs = output)
model.summary()


plot_model(model, to_file='model_2.png')



#hist = pd.read_csv('./Histograms/'+os.listdir('Histograms')[0], usecols = ['0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8','0.8-0.9','0.9-1.0'])
#data_hist = np.array(hist)

#activations = []
#for j in os.listdir('./ROI_conv2_1/'+os.listdir('Histograms')[0][:-4]):
#    activation = np.load('./ROI_conv2_1/'+os.listdir('Histograms')[0][:-4]+'/'+j)
#    activations.append(activation)
    
#data_activations = np.array(activations)

#print(data_hist.shape)
#print(data_activations.shape)

#for i in tqdm(range(1, len(os.listdir('Histograms')))):
#    hist = pd.read_csv('./Histograms/'+os.listdir('Histograms')[i], usecols = ['0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8','0.8-0.9','0.9-1.0'])
#    hist = np.array(hist)
   
#    if hist.shape != (0, 9):
#        data_hist = np.append(data_hist, hist, axis = 0)
    
#    activations = []
#    for j in os.listdir('./ROI_conv2_1/'+os.listdir('Histograms')[i][:-4]):
#        try:
#            activation = np.load('./ROI_conv2_1/'+os.listdir('Histograms')[i][:-4]+'/'+j)
#            activations.append(activation)
#        except:
#            pass
        
#    activations = np.array(activations)
    
#    if activations.shape != (0,):
#        data_activations = np.append(data_activations, activations, axis = 0)
    
#print(data_hist.shape)
#print(data_activations.shape)

#np.save('data_hist.npy', data_hist)
#np.save('data_activations_conv2_1.npy', data_activations)

data_hist = np.load('data_hist.npy')
data_activations = np.load('data_activations.npy')
data_activations_conv2_1 = np.load('data_activations_conv2_1.npy')

sc_hist = StandardScaler()
X_train_hist = sc_hist.fit_transform(data_hist)


data_y = pd.read_csv('train_cow_dataset.csv', usecols = ['target'])
print(data_y.shape)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit([data_activations, X_train_hist, data_activations_conv2_1], data_y, validation_split = 0.1, batch_size = 128, epochs = 20, shuffle = True)

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




hist = pd.read_csv('./Histograms_val/'+os.listdir('Histograms_val')[0], usecols = ['0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8','0.8-0.9','0.9-1.0'])
test_data_hist = np.array(hist)

activations = []
for j in os.listdir('./ROI_conv2_1_val/'+os.listdir('Histograms_val')[0][:-4]):
    activation = np.load('./ROI_conv2_1_val/'+os.listdir('Histograms_val')[0][:-4]+'/'+j)
    activations.append(activation)
    
test_data_activations_conv2_1 = np.array(activations)

print(test_data_hist.shape)
print(test_data_activations_conv2_1.shape)


for i in tqdm(range(1, len(os.listdir('Histograms_val')))):
    hist = pd.read_csv('./Histograms_val/'+os.listdir('Histograms_val')[i], usecols = ['0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8','0.8-0.9','0.9-1.0'])
    hist = np.array(hist)
  
    if hist.shape != (0, 9):
        test_data_hist = np.append(test_data_hist, hist, axis = 0)
    
    activations = []
    for j in os.listdir('./ROI_conv2_1_val/'+os.listdir('Histograms_val')[i][:-4]):
        try:
            activation = np.load('./ROI_conv2_1_val/'+os.listdir('Histograms_val')[i][:-4]+'/'+j)
            activations.append(activation)
        except:
            pass
        
    activations = np.array(activations)
   
    if activations.shape != (0,):
        test_data_activations_conv2_1 = np.append(test_data_activations_conv2_1, activations, axis = 0)
   
print(test_data_hist.shape)
print(test_data_activations.shape)
print(test_data_activations_conv2_1.shape)

X_test_hist = sc_hist.transform(test_data_hist)

predictions = model.predict([test_data_activations, X_test_hist, test_data_activations_conv2_1], verbose = True)

print(predictions.shape)

test_df = pd.DataFrame(predictions, columns = ['predictions'])
test_df.to_csv('predictions_2.csv')