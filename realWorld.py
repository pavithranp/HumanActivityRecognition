import csv
import numpy as np
import tensorflow as tf
import pandas as pd
tf.get_logger().setLevel('INFO')

activities = ['climbingdown', #1
 'climbingup',#2
 'jumping',#3
 'lying',#4
 'running',#5
 'sitting',#6
 'standing',#7
 'walking']#8
# uncomment to train the required datasets, caution it takes very long time to train more than 4-5 users
users =['proband1',
        'proband3',
        #'proband5',
        #'proband9',
        #'proband10',
        #'proband11',
        #'proband12',
        #'proband15'
        ]
sensor_position = ['chest','forearm','head','shin','thigh','upperarm','waist']
complete_data=[]
complete_label=[]
windowsize=250

path = "RealWorldData/"

def get_user_data(user,windowsize):
  for i,activity in enumerate(activities):
    #print(activity)
    fulldata =[]
    minsize=999999

    for pos in sensor_position:
      with open(path+user+"/data/acc_"+activity+"_csv/acc_"+activity+"_"+pos+".csv") as csvfile:
        accdata =  np.genfromtxt(csvfile, delimiter=',')
        fulldata.append(accdata)
        #print(accdata.shape)
        if(accdata.shape[0]<minsize):
          minsize=accdata.shape[0]

      with open(path+user+"/data/gyr_"+activity+"_csv/Gyroscope_"+activity+"_"+pos+".csv") as csvfile:
        gyrdata =  np.genfromtxt(csvfile, delimiter=',')
        fulldata.append(gyrdata)
        if(gyrdata.shape[0]<minsize):
          minsize=gyrdata.shape[0]
    minsize=int(minsize/1000)*1000
    #print(minsize)
    full_data = [ x[1:minsize+1,2:5] for x in fulldata ]
    full_data =np.column_stack(tuple(full_data))

    batched_label=[i+1]*int(minsize/windowsize)
    batched_data =np.split(full_data,minsize/windowsize)
    complete_data.extend(batched_data)
    complete_label.extend(batched_label)
    #full_data =np.column_stack((fulldata[0][1:minsize,2:5],fulldata[1][1:minsize,2:5],fulldata[2][1:minsize,2:5],fulldata[3][1:minsize,2:5],fulldata[4][1:minsize,2:5],fulldata[5][1:minsize,2:5],fulldata[6][1:minsize,2:5]))
  return complete_data,complete_label


complete_data=[]
complete_label=[]
for user in users:
  print(user)
  data,label=get_user_data(user,windowsize)   
  print(np.shape(data))

complete_data = np.array(complete_data)
complete_label = np.array(complete_label)
size=complete_data.shape[0]
complete_ds=tf.data.Dataset.from_tensor_slices((complete_data,complete_label)).shuffle(size)
train_size = int(0.7 * size)
val_size = int(0.15 * size)
test_size = int(0.15 * size)


train_ds = complete_ds.take(train_size).batch(10)
val_ds = complete_ds.skip(train_size).batch(10)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM,Dense,AveragePooling1D,Flatten,TimeDistributed,Dropout
model = Sequential()
model.add(LSTM(250, input_shape=(250, 42),return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(9, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
checkpoint_path="weights1-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=False)
model.fit(train_ds, epochs=100, verbose=1,validation_data=val_ds,use_multiprocessing=True,callbacks=[checkpoint])
