from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Flatten, Dropout
from keras.models import Model, Sequential
from read import get_data
from save import save_history
from utils import print_score


train_generator, val_generator, image_shape, n_classes = get_data()

input_layer = Input(shape=image_shape)

x = Conv2D(32, (4, 4), activation='relu', padding='same')(input_layer)
x = Conv2D(32, (4, 4), activation='relu', padding='same')(x)
x = MaxPool2D((2, 2), strides=(2, 2))(x)

x = Conv2D(64, (4, 4), activation='relu', padding='same')(x)
x = Conv2D(64, (4, 4), activation='relu', padding='same')(x)
x = MaxPool2D((2, 2), strides=(2, 2))(x)

x = Conv2D(128, (4, 4), activation='relu', padding='same')(x)
x = Conv2D(128, (4, 4), activation='relu', padding='same')(x)
x = MaxPool2D((2, 2), strides=(2, 2))(x)

x = Conv2D(256, (4, 4), activation='relu', padding='same')(x)
x = Conv2D(256, (4, 4), activation='relu', padding='same')(x)
# x = MaxPool2D((2,2), strides=(2, 2))(x)

x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
y = Dense(n_classes, activation='softmax')(x)

net = Model(input_layer, y)

net.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])


history = net.fit_generator(train_generator,
                  epochs=100,
                  steps_per_epoch=100,
                  verbose=1,
                  validation_data=val_generator)

print_score(net, train_generator, val_generator)

save_history(history, 'history_3.pk')
net.save('model_3.h5')

del net