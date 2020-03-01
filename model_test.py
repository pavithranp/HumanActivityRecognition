import tensorflow as tf
import pathlib
import tensorflow.keras as k
import numpy as np
root_path = 'Dataset/HAPT Data Set/'

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
           currentActivity=int(line[2])
           startIndex=int(line[3])
           endIndex=int(line[4])
           with open(root_path+datasettype+accel_filename) as acc_file:
             for idx,i in enumerate(acc_file):
               i=i.split(' ')
               if idx>startIndex and idx<endIndex:
                 ax.append(i[0])
                 ay.append(i[1])
                 az.append(i[2])
                 label.append(currentActivity)
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


tax,tay,taz,tbx,tby,tbz,tlabel = generate_data_arrays(test_label_file,"Test/")

test_dataset=np.transpose(np.array([tax,tay,taz,tbx,tby,tbz,tlabel]))

def checkList(lst):
    return len(set(lst)) == 1
def input_windowing(dataset,window_size=250,shift=125):
  data = []
  labels = []
  index = 0
  length = len(dataset)
  slices = int(length/shift)
  for i in range(slices-1):
   if checkList(dataset[i*shift:((i*shift) +window_size ),6]) == 1:  #optional task
    data.append(dataset[i*shift:((i*shift) +window_size ),0:6])
    labels.append(dataset[i*shift+1,6]) #optional task
    #labels.append(dataset[i*shift:((i*shift) +window_size ),6]) #required task
  return np.array(data), np.array(labels)
x_test,y_test = input_windowing(test_dataset)

print(np.shape(x_test))
print(np.shape(y_test))
test_ds=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_ds=test_ds.cache().batch(20)

new_model = tf.keras.models.load_model('seq2label.hdf5')


l=new_model.predict(x_test)
g = [np.argmax(i) for i in l]
#print(g)
#print(y_test)
con_mat=tf.math.confusion_matrix(y_test,g)
with tf.Session():
   print('Confusion Matrix: \n\n', tf.Tensor.eval(con_mat,feed_dict=None, session=None))

