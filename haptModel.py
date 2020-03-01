# -*- coding: utf-8 -*-
"""HAPT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13HuBVjL2pQmV3tPASjjze738D5LGGTRb
"""

#from google.colab import drive
#drive.mount ('/content/gdrive' )
root_path = 'Dataset/HAPT Data Set/'

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
import pathlib
import tensorflow.keras as k
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.contrib.data.python.ops import sliding

train_directory = root_path + "Training/"
test_directory = root_path + "Test/"
validation_directory = root_path + "Validation/"
train_label_file = root_path+"train.txt"
validation_label_file = root_path+"eval.txt"
test_label_file = root_path+"test.txt"
print(train_directory)
train_data_dir = pathlib.Path(train_directory)
train_image_count = len(list(train_data_dir.glob("*.txt")))
'''for x in list(train_data_dir.glob("*.txt")):
  print(x)'''
x=list(train_data_dir.glob("*.txt"))
print(x[0])

#Normalised dataset arrays
def generate_data_arrays(filewithpath,datasettype):
    ax=[]
    ay=[]
    az=[]
    bx=[]
    by=[]
    bz=[]
    label=[]
    i=1
    with open(filewithpath) as data_file:
        for line in data_file:
           if len(line.strip()) == 0 :
             print("empty line")
             break;
           line=line.split(' ')
           if int(line[0]) < 10:
             line[0] = "0"+ line[0]
           if int(line[1]) < 10:
             line[1] = "0"+ line[1] 
           accel_filename = "acc_exp"+line[0]+"_user"+line[1]+".txt"
           gyro_filename = "gyro_exp"+line[0]+"_user"+line[1]+".txt"
           #print("reading file... "+accel_filename)
           currentActivity=float(line[2])
           startIndex=int(line[3])
           endIndex=int(line[4])
           with open(root_path+datasettype+accel_filename) as acc_file:
             for idx,i in enumerate(acc_file):
               i=i.split(' ')
               if idx>startIndex and idx<endIndex:
                 ax.append(i[0])
                 ay.append(i[1])
                 az.append(i[2])
                 label.append((currentActivity-6.0)/12.0)
           with open(root_path+datasettype+gyro_filename) as gyro_file:
             for idx,i in enumerate(gyro_file):
               i=i.split(' ')
               if idx>startIndex and idx<endIndex:
                 bx.append(i[0])
                 by.append(i[1])
                 bz.append(i[2])
        ax=np.array(ax, dtype=np.float32)
        ay=np.array(ay, dtype=np.float32)
        az=np.array(az, dtype=np.float32)
        bx=np.array(bx, dtype=np.float32)
        by=np.array(by, dtype=np.float32)
        bz=np.array(bz, dtype=np.float32)
           #if int(line[0])==2:
            # break
    print("made lists for "+datasettype)
    return ax,ay,az,bx,by,bz,label

#load data to arrays 
#takes some time
ax,ay,az,bx,by,bz,label = generate_data_arrays(train_label_file,"Training/")
vax,vay,vaz,vbx,vby,vbz,vlabel = generate_data_arrays(validation_label_file,"Validation/")
tax,tay,taz,tbx,tby,tbz,tlabel = generate_data_arrays(test_label_file,"Test/")

train_dataset=np.transpose(np.array([ax,ay,az,bx,by,bz,label]))
validation_dataset=np.transpose(np.array([vax,vay,vaz,vbx,vby,vbz,vlabel]))
test_dataset=np.transpose(np.array([tax,tay,taz,tbx,tby,tbz,tlabel]))

#output_train_ds = tf.data.Dataset.from_tensor_slices(label)
#train_ds = tf.data.Dataset.zip((input_train_ds, output_train_ds)).window(128)

train_dataset = np.array(train_dataset)

print(np.shape(train_dataset))
print(train_dataset[0:1:,6])
WINDOW_SIZE=100
def checkList(lst): 
    return len(set(lst)) == 1
def input_windowing(dataset,window_size=100,shift=125):
  data = []
  labels = []
  index = 0
  length = len(dataset)
  slices = int(length/shift)
  for i in range(slices-1):
    if checkList(dataset[i*shift:((i*shift) +window_size ),6]) == 1:  #optional task
      data.append(dataset[i*shift:((i*shift) +window_size ),0:6])
      #labels.append(dataset[i*shift+1,6])
      labels.append(dataset[i*shift:((i*shift) +window_size ),6])
  return np.array(data), np.array(labels)

x_train,y_train=input_windowing(train_dataset)
x_val,y_val=input_windowing(validation_dataset)
x_test,y_test = input_windowing(test_dataset)

np.shape(x_train)

#Fetch the sensor data for each individual labels and plot the graph

#To fetch the desired label data. Here data is fetched for label 12 - LIE_TO_STAND
count = 0
for i,elem in enumerate(y_train):
  for y in elem:
    if int(y*12+6) == 12:
      print(i)
      break;
  count = count +1
  continue
print(count)
#Sensor data plots
#Plot the accelerometer and gyroscope values with label - 12 (LIE_TO_STAND)
plt.plot(x_train[74])
plt.title('Sensor data plot for label 12 - LIE_TO_STAND')
plt.ylabel('Sensor values')
plt.xlabel('Samples')
plt.legend(['ax', 'ay', 'az', 'gx','gy','gz'], loc='upper left')
plt.show()

#n_timesteps, n_features, n_outputs = input_train_ds.shape[1], input_train_ds.shape[2], input_train_ds.shape[1]
	# define model
train_ds =tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(20).shuffle(4418)
validation_ds =tf.data.Dataset.from_tensor_slices((x_val,y_val)).batch(20).shuffle(685)
test_ds =tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(20)
#for subs in train_ds.take(1):
 # print(subs)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM,Dense,AveragePooling1D,Flatten,TimeDistributed

model = Sequential()
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(32, return_sequences=True,input_shape=(WINDOW_SIZE,6)))
model.add(tf.keras.layers.LSTM(16, activation='relu'))
model.add(tf.keras.layers.Dense(WINDOW_SIZE))
model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
model.summary()

'''model = Sequential()
model.add(LSTM(100, input_shape=(250, 6)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(13, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
''' 

filepath="/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint_path = root_path + filepath

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='max',save_weights_only=False)
callbacks_list = [checkpoint]
model.fit(train_ds, epochs=150, verbose=1,validation_data=validation_ds,callbacks=[checkpoint],use_multiprocessing=True)

l=model.predict(x_test)

np.shape(l)
#verification
print(l[98]*12 +6)
print(y_test[98]*12 + 6)
