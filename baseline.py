from keras.layers import Convolution2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten
from keras import Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_model():
  model = Sequential()

  model.add(Convolution2D(64, (3, 3), padding='same', input_shape=(48,48,1)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
  model.add(Dropout(0.25))

  model.add(Convolution2D(128, (5, 5), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
  model.add(Dropout(0.25))

  model.add(Flatten())

  model.add(Dense(256))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.25))

  model.add(Dense(7))
  model.add(Activation('softmax'))

  return model

if __name__ == '__main__':
  EPOCHS = 150

  ######################################
  # creating training and testing dataset
  ######################################
  train_path = "fer2013/train/"
  test_path = "fer2013/test/"

  train_datagen = ImageDataGenerator(rescale=1./255)
  training_data = train_datagen.flow_from_directory(train_path,
                                                    target_size=(48,48),
                                                    shuffle=True,
                                                    color_mode="grayscale",
                                                    class_mode="categorical")
  test_datagen = ImageDataGenerator(rescale=1./255)
  testing_data = test_datagen.flow_from_directory(test_path,
                                                  target_size=(48,48),
                                                  shuffle=True,
                                                  color_mode="grayscale",
                                                  class_mode="categorical")

  ######################################
  # baseline model
  ######################################
  baseline = get_model()
  baseline.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

  ######################################
  # training
  ######################################
  history = baseline.fit_generator(
    generator=training_data,
    shuffle=True,
    epochs=EPOCHS,
    use_multiprocessing=False
  )
  
  ######################################
  # evaluation
  ######################################
  test_result = baseline.evaluate_generator(testing_data)
  print("test loss, test acc:", test_result)

  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'dev'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'dev'], loc='upper left')
  plt.show()
  