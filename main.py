from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix as cf
import pandas as pd
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Activation,Dropout,Flatten,Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
import pandas as pd
import numpy as np 
import pandas as pd
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Activation,Dropout,Flatten,Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
data = pd.read_csv('E:\C\Cadence\SPB_Data/fer2013.csv')
num_of_instances = len(data) #获取数据集的数量
print("数据集的数量为：",num_of_instances)
def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(filters = F1, kernel_size= (1, 1), strides = (s,s),padding="valid", name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    

    # Second component of main path 
    X = Conv2D(filters = F2, kernel_size=(f,f), strides=(1,1), name = conv_name_base + '2b', padding="same",kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name= bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(filters = F3, kernel_size=(1,1), strides = (1,1), name= conv_name_base + '2c',padding="valid", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(filters = F3, kernel_size= (1,1), strides=(s,s), name=conv_name_base + '1', padding="valid", kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base+'1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X = Add()([X_shortcut,X])
    X = Activation("relu")(X)
    
    
    return X

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    
    # Second component of main path 
    X = Conv2D(filters = F2, kernel_size=(f,f), strides = (1,1), padding='same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(filters = F3, kernel_size=(1,1), strides = (1,1), padding="valid", name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet(input_shape=(48,48,1),classes=7):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(48, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = convolutional_block(X, f = 3, filters = [48, 48, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [48, 48, 256], stage=2, block='b')
    X = identity_block(X, 3, [48, 48, 256], stage=2, block='c')
    X = AveragePooling2D((2,2), name='avg_pool')(X)
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    model = Model(inputs = X_input, outputs = X, name='ResNet')
    return model

pixels = data['pixels']
emotions = data['emotion']
usages = data['usage']

num_classes = 7   #表情的类别数目
x_train,y_train,x_test,y_test = [],[],[],[]

for emotion,img,usage in zip(emotions,pixels,usages):    
    try: 
        emotion = keras.utils.to_categorical(emotion,num_classes)   # 独热向量编码
        val = img.split(" ")
        pixels = np.array(val,'float32')
        
        if(usage == 'Training'):
            x_train.append(pixels)
            y_train.append(emotion)
        elif(usage == 'PublicTest'):
            x_test.append(pixels)
            y_test.append(emotion)
    except:
        print("",end="")

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = x_train.reshape(-1,48,48,1)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_test = x_test.reshape(-1,48,48,1)
for i in range(4): 
    plt.subplot(221+i)
    plt.gray()
    plt.imshow(x_train[i].reshape([48,48]))
    
Data = pd.read_csv(r"E:\C\Cadence\SPB_Data\fer2023.csv")

Pixels = Data['pixels']
Emotions = Data['emotion']
Usages = Data['usage']

num_classes = 7   #表情的类别数目
X_train,Y_train,X_test,Y_test = [],[],[],[]

for Emotion,Img,Usage in zip(Emotions,Pixels,Usages):    

    Emotion = keras.utils.to_categorical(Emotion,num_classes)   # 独热向量编码
    Val = Img.split(" ")
    Pixels = np.array(Val,'float32')
    X_train.append(Pixels)
        
       
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_train = X_train.reshape(-1,48,48,1)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
X_test = X_test.reshape(-1,48,48,1)

batch_size = 8
epochs = 20

model = Sequential()

#第一层卷积层
model.add(Conv2D(input_shape=(48,48,1),filters=32,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

#第二层卷积层
model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

#第三层卷积层
model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Flatten())

#全连接层
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(7,activation = 'softmax'))

#进行训练
#model.compile(loss = 'categorical_crossentropy',optimizer = Adam(),metrics=['accuracy'])

model1 = ResNet(input_shape = (48, 48, 1), classes = 7)
model1.compile(loss='categorical_crossentropy',optimizer = Adam(), metrics=['accuracy'])
model1.fit(x_train,y_train,batch_size=batch_size,epochs=epochs)
y_pred=model1.predict(x_train)
cm=cf(y_train.argmax(axis=1),y_pred.argmax(axis=1))
#plot_confusion_matrix(cm,title='Confusion Matrix')
print(cm)
plt.figure()
plt.imshow(cm,interpolation='nearest')
train_score = model1.evaluate(x_train, y_train, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', 100*train_score[1])
 
test_score = model1.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', 100*test_score[1])
y_test_pred = model.predict(X_train,verbose=1)
y_test_pred = np.argmax(y_test_pred, axis=1)
print(y_test_pred)