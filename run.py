from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from glob import glob


IMAGE_SIZE = [100, 100]
EPOCHS = 1
BATCH_SIZE = 32

train_path = './large_files/fruits-360-small/Training'
valid_path = './large_files/fruits-360-small/Validation'

image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')

folders = glob(train_path + '/*')

vgg = VGG16(
    input_shape=IMAGE_SIZE + [3],
    weights='imagenet',
    include_top=False
)

for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)
model.summary()
model.compile(
  loss='categorical_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy']
)

gen = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  preprocessing_function=preprocess_input
)

train_gen = gen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=BATCH_SIZE
)

test_gen = gen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=BATCH_SIZE
)

r = model.fit_generator(
    generator=train_gen,
    validation_data=test_gen,
    epochs=EPOCHS,
    steps_per_epoch=len(image_files)//BATCH_SIZE,
    validation_steps=len(valid_image_files)//BATCH_SIZE
)

plt.plot(r.history['loss'], label='Training Loss')
plt.plot(r.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()