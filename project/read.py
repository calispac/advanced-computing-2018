from keras.preprocessing.image import ImageDataGenerator
import pickle
import os
import numpy as np


def get_data():

    path_train = 'asl_alphabet_train/'
    path_test = 'asl_alphabet_test/'
    n_classes = 29
    target_size = (128, 128)  # all images are rescaled to this size
    image_shape = target_size + (3, )
    batch_size = 64
    color_mode = 'rgb'

    data_augmentor = ImageDataGenerator(rescale=1/255, )

    data_augmentor_validation = ImageDataGenerator(rescale=1/255, )

    train_generator = data_augmentor.flow_from_directory(path_train,
                                                         target_size=target_size,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         color_mode=color_mode)

    val_generator = data_augmentor_validation.flow_from_directory(path_test,
                                                         target_size=target_size,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         color_mode=color_mode)

    return train_generator, val_generator, image_shape, n_classes


def get_new_data(path):

    target_size = (128, 128)  # all images are rescaled to this size
    batch_size = 1
    color_mode = 'rgb'

    data_augmentor = ImageDataGenerator(rescale=1/255, )

    generator = data_augmentor.flow_from_directory(path, target_size=target_size,
                                                         batch_size=batch_size,
                                                         color_mode=color_mode,
                                                        shuffle=False)
    return generator


def load_history(filename):

    with open(filename, 'rb') as file:

        return pickle.load(file)


def create_validation_dataset(split_ratio=0.1):

    train_path = 'asl_alphabet_train/'
    test_path = 'asl_alphabet_test/'
    label_names = os.listdir(train_path)

    for label_name in label_names:

        input_path = os.path.join(train_path, label_name)
        output_path = os.path.join(test_path, label_name)

        if os.path.exists(output_path):

            exit()

        else:

            os.mkdir(output_path)

        files = os.listdir(input_path)
        n_files = len(files)
        n_validation_files = int(n_files * split_ratio)

        file_indices = np.arange(n_files, dtype=int)
        np.random.shuffle(file_indices)
        file_indices = file_indices[:n_validation_files]

        for index in file_indices:

            file = files[index]
            input_file_path = os.path.join(input_path, file)
            output_file_path = os.path.join(output_path, file)
            print("Moving file : {} to {}".format(input_file_path,
                                                  output_file_path))
            os.rename(input_file_path, output_file_path)

    return


if __name__ == '__main__':

    # create_validation_dataset(0.1)

    generator = get_new_data()

    for image, label in generator:

        print(image.max())

    pass


