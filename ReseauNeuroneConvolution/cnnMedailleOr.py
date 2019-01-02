

# Partie 1 - Contruction du CNN

# Importation des modules
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initialiser le CNN
classifier = Sequential()

# Etape 1 - Convolution
classifier.add(Convolution2D(filters=32, kernel_size=3, strides=1,
                             input_shape=(150,150,3), 
                             activation="relu"))

# Etape 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Ajout 2eme couche de convolution et pooling
classifier.add(Convolution2D(filters=32, kernel_size=3, strides=1,
                             activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))
# Ajout 3eme couche de convolution et pooling
classifier.add(Convolution2D(filters=64, kernel_size=3, strides=1,
                             activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))
# Ajout 4eme couche de convolution et pooling
classifier.add(Convolution2D(filters=64, kernel_size=3, strides=1,
                             activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Etape 3 -Flattening
classifier.add(Flatten())

# Etape 4 - reseau de neurone artificiel complètement connecté
    # couche caché
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dropout(0.3))
    # couche sortie
classifier.add(Dense(units=1,activation="sigmoid"))

# Compilation
classifier.compile(optimizer="adam", loss="binary_crossentropy", 
                   metrics=["accuracy"])

# Entrainer le CNN sur nos images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=250,
        epochs=75,
        validation_data=test_set,
        validation_steps=63)

# acc training 0.8587
# acc test 0.7660

# acc training 0.8636
# acc test 0.8130

classifier.save('modelOr1.h5')



from keras.models import load_model
model = load_model('modelOr1.h5')

from keras.preprocessing import image
import numpy as np
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
classes = model.predict_classes(test_image, batch_size=10)
if classes[0][0] == 1:
    prediction = "chien"
else:
    prediction = "chat"

test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
classes = model.predict_classes(test_image, batch_size=10)
if classes[0][0] == 1:
    prediction = "chien"
else:
    prediction = "chat"

