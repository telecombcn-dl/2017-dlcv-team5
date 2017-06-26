from __future__ import print_function
import os
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input, Activation
from keras.utils import to_categorical, normalize
from keras import applications
from keras.optimizers import SGD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = '/imatge/mcata/Terrassa/bottleneck_fc_model.h5'
dir = "/imatge/mcata/Terrassa/"
train_data_dir = '/imatge/mcata/Terrassa/train'
validation_data_dir = '/imatge/mcata/Terrassa/val'
nb_train_samples = 450
nb_validation_samples = 180
epochs = 20
batch_size = 20

# Load dict images
def load_image(dir, type):
    full_dir = dir + type + 'images/'
    dict_images = {}
    for filename in os.listdir(full_dir):
        name = os.path.splitext(filename)[0]
        img = load_img(full_dir + filename, target_size=(img_width, img_height))
        dict_images[name] = img_to_array(img, data_format='channels_last')

    return dict_images

# Load dict labels
def load_labels(dir, type):
    dict_labels = {}
    classes = set([])
    full_dir = dir + type
    with open(full_dir + 'annotation.txt', 'r') as f:
        next(f)
        for l in f:
            x, y = l.split()
            dict_labels[x] = y
            classes.add(y)

    return classes, dict_labels

def create_db(dict_img, dict_labels, classes):
    x_train = []
    labels = []
    for key, value in dict_img.iteritems():
        x_train.append(value)
        name = dict_labels[key]
        idx = classes.index(name)
        labels.append(idx)

    x_train = np.asarray(x_train)
    y_train = np.asarray(labels)

    return x_train, y_train



# Generate a model with all layers (with top)
vgg16 = applications.VGG16(weights='imagenet', include_top=True)

#Add a layer where input is the output of the  second last layer
x = Dense(13, activation='softmax', name='predictions')(vgg16.layers[-2].output)

#Then create the corresponding model
model = Model(input=vgg16.input, output=x)
model.summary()
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])
print('model compiled')

# model = applications.VGG16(include_top=True, weights='imagenet', input_shape=(img_width, img_height, 3))
# print('model loaded')
# model2 = Sequential()
# model2.add(model)
# model2.layers.pop()
# print (model2.output_shape)
# print('ups')
# model2.outputs = [model2.layers[-1].output]
# model2.layers[-1].outbound_nodes = []
# model2.add(Dense(13, activation='softmax'))
# print(model.output_shape)



dict_img = load_image(dir, 'train/')
print("# train samples:", len(dict_img))

classes, dict_labels = load_labels(dir, 'train/')
classes = list(classes)
print('Classes:', classes)
print("# train labels:", len(dict_labels))

x_train, y_train = create_db(dict_img, dict_labels, classes)
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

dict_img = load_image(dir, 'val/')
print("# val samples:", len(dict_img))

classes, dict_labels = load_labels(dir, 'val/')
classes = list(classes)
print('Classes:', classes)
print("# val labels:", len(dict_labels))
num_classes = len(classes)
x_val, y_val = create_db(dict_img, dict_labels, classes)
print('x_val shape:', x_val.shape)
print('y_val shape:', y_val.shape)
#y_train = np.array(y_train)
#y_val = np.array(y_val)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_val /= 255
y_train = to_categorical(y_train, 13)
y_val = to_categorical(y_val, 13)


history = model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(x_val, y_val))
print('top model trained')

model.save_weights(top_model_weights_path)
# list all data in history
print(history.history.keys())
fig, axis = plt.subplots(1,2)#,figsize=(15,5))

# summarize history for accuracy

axis[0].plot(history.history['acc'])
axis[0].plot(history.history['val_acc'])
axis[0].set_title('model accuracy')
axis[0].set_ylabel('accuracy')
axis[0].set_xlabel('epoch')
axis[0].legend(['train', 'test'], loc='upper left')
#plt.show()
#axis axis[0].savefig('/imatge/mcata/Terrassa/accuracy_22.png')
# summarize history for loss

axis[1].plot(history.history['loss'])
axis[1].plot(history.history['val_loss'])
axis[1].set_title('model loss')
axis[1].set_ylabel('loss')
axis[1].set_xlabel('epoch')
axis[1].legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('/imatge/mcata/Terrassa/task_3.png')

