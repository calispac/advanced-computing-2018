from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Flatten, Dropout
from keras.models import Model, Sequential
from read import get_data
from save import save_history
from utils import print_score


train_generator, val_generator, image_shape, n_classes = get_data()

net = Sequential()
net.add(Conv2D(64, kernel_size=4, strides=1, activation='relu',
               input_shape=image_shape))
net.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
net.add(Dropout(0.5))
net.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))
net.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))
net.add(Dropout(0.5))
net.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))
net.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))
net.add(Flatten())
net.add(Dropout(0.5))
net.add(Dense(512, activation='relu'))
net.add(Dense(n_classes, activation='softmax'))
net.compile(optimizer='adam', loss='categorical_crossentropy',
                 metrics=["accuracy"])

history = net.fit_generator(train_generator,
                            epochs=100,
                            steps_per_epoch=100,
                            verbose=1,
                            validation_data=val_generator)

print_score(net, train_generator, val_generator)

save_history(history, 'history_2.pk')

net.save('model_2.h5')

del net