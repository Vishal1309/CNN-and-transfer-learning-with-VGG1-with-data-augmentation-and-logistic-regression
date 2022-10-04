import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys


class BaselineCNN():

    def init(self):

        model = keras.Sequential()
        # block 1
        model.add(Conv2D(32, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
        model.add(MaxPooling2D((2, 2)))
        # block 2
        model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        # block 3
        model.add(Conv2D(128, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu',
                  kernel_initializer='he_uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    # plot diagnostic learning curves
    def summarize_diagnostics(self, history, filename):
        # plot loss
        plt.subplot(211)
        plt.title('Cross Entropy Loss')
        plt.plot(history.history['loss'], color='blue', label='train')
        plt.plot(history.history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(212)
        plt.title('Classification Accuracy')
        plt.plot(history.history['accuracy'], color='blue', label='train')
        plt.plot(history.history['val_accuracy'], color='orange', label='test')
        plt.tight_layout()
        plt.legend()
        # save plot to file
        plt.savefig(filename + '.png')
        plt.close()

    # run the test harness for evaluating a model
    def run_test_harness(self, filename, epochs=20, dataset_home='preprocessed_dataset_vultures_vs_sharks/', verbose=1):
        # define model
        model = self.init()
        # create data generator
        datagen = ImageDataGenerator(rescale=1.0/255.0)
        # prepare iterator
        train_it = datagen.flow_from_directory(dataset_home+'train/',
                                               class_mode='binary', batch_size=5, target_size=(200, 200))
        test_it = datagen.flow_from_directory(dataset_home+'test/',
                                              class_mode='binary', batch_size=5, target_size=(200, 200))
        # fit model
        history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
                                      validation_data=test_it, validation_steps=len(test_it), epochs=epochs, verbose=verbose)
        # evaluate model
        _, acc = model.evaluate_generator(
            test_it, steps=len(test_it), verbose=0)
        print('> %.3f' % (acc * 100.0))
        # learning curves
        self.summarize_diagnostics(history, filename)


class VGG1():
    def init(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu',
                  kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu',
                  kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model


class VGG1_transfer():
    def init(self):
        # load model
        vgg1_obj = VGG1()
        model = vgg1_obj.init()
        # mark loaded layers as not trainable
        for layer in model.layers:
            layer.trainable = False
        # add new classifier layers
        flat1 = Flatten()(model.layers[-1].output)
        class1 = Dense(128, activation='relu',
                       kernel_initializer='he_uniform')(flat1)
        output = Dense(1, activation='sigmoid')(class1)
        # define new model
        model = Model(inputs=model.inputs, outputs=output)
        # compile model
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def summarize_diagnostics(self, history, filename):
        # plot loss
        plt.subplot(211)
        plt.title('Cross Entropy Loss')
        plt.plot(history.history['loss'], color='blue', label='train')
        plt.plot(history.history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(212)
        plt.title('Classification Accuracy')
        plt.plot(history.history['accuracy'], color='blue', label='train')
        plt.plot(history.history['val_accuracy'], color='orange', label='test')
        plt.tight_layout()
        plt.legend()
        # save plot to file
        plt.savefig(filename + '.png')
        plt.close()

    # run the test harness for evaluating a model
    def run_test_harness(self, filename, epochs=20, dataset_home='preprocessed_dataset_vultures_vs_sharks/', verbose=1):
        # define model
        model = self.init()
        # create data generator
        datagen = ImageDataGenerator(rescale=1.0/255.0)
        # prepare iterators
        train_it = datagen.flow_from_directory(dataset_home+'train/',
                                               class_mode='binary', batch_size=5, target_size=(200, 200))
        test_it = datagen.flow_from_directory(dataset_home+'test/',
                                              class_mode='binary', batch_size=5, target_size=(200, 200))
        # fit model
        history = model.fit(train_it, steps_per_epoch=len(train_it),
                            validation_data=test_it, validation_steps=len(test_it), epochs=epochs, verbose=verbose)
        # evaluate model
        _, acc = model.evaluate(
            test_it, steps=len(test_it), verbose=0)
        print('> %.3f' % (acc * 100.0))
        # learning curves
        self.summarize_diagnostics(history, filename)

    # run the test harness for evaluating a model
    def run_test_harness_augmented(self, filename, epochs=20, dataset_home='preprocessed_dataset_vultures_vs_sharks/', verbose=1):
        # define model
        model = self.init()
        # create data generators
        train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                           width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1.0/255.0)
        # prepare iterators
        train_it = train_datagen.flow_from_directory(dataset_home+'train/',
                                                     class_mode='binary', batch_size=5, target_size=(200, 200))
        test_it = test_datagen.flow_from_directory(dataset_home+'test/',
                                                   class_mode='binary', batch_size=5, target_size=(200, 200))
        # fit model
        history = model.fit(train_it, steps_per_epoch=len(train_it),
                            validation_data=test_it, validation_steps=len(test_it), epochs=epochs, verbose=verbose)
        # evaluate model
        _, acc = model.evaluate(
            test_it, steps=len(test_it), verbose=0)
        print('> %.3f' % (acc * 100.0))
        # learning curves
        self.summarize_diagnostics(history, filename)


class VGG16_transfer():
    def init(self):
        model = VGG16(include_top=False, input_shape=(200, 200, 3))
        # mark loaded layers as not trainable
        for layer in model.layers:
            layer.trainable = False
        # add new classifier layers
        flat1 = Flatten()(model.layers[-1].output)
        class1 = Dense(128, activation='relu',
                       kernel_initializer='he_uniform')(flat1)
        output = Dense(1, activation='sigmoid')(class1)
        # define new model
        model = Model(inputs=model.inputs, outputs=output)
        # compile model
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def summarize_diagnostics(self, history, filename):
        # plot loss
        plt.subplot(211)
        plt.title('Cross Entropy Loss')
        plt.plot(history.history['loss'], color='blue', label='train')
        plt.plot(history.history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(212)
        plt.title('Classification Accuracy')
        plt.plot(history.history['accuracy'], color='blue', label='train')
        plt.plot(history.history['val_accuracy'], color='orange', label='test')
        plt.tight_layout()
        plt.legend()
        # save plot to file
        plt.savefig(filename + '.png')
        plt.close()

    # run the test harness for evaluating a model
    def run_test_harness(self, filename, epochs=20, dataset_home='preprocessed_dataset_vultures_vs_sharks/', verbose=1):
        # define model
        model = self.init()
        # create data generator
        datagen = ImageDataGenerator(featurewise_center=True)
        # specify imagenet mean values for centering
        datagen.mean = [123.68, 116.779, 103.939]
        # prepare iterators
        train_it = datagen.flow_from_directory(dataset_home+'train/',
                                               class_mode='binary', batch_size=5, target_size=(200, 200))
        test_it = datagen.flow_from_directory(dataset_home+'test/',
                                              class_mode='binary', batch_size=5, target_size=(200, 200))
        # fit model
        history = model.fit(train_it, steps_per_epoch=len(train_it),
                            validation_data=test_it, validation_steps=len(test_it), epochs=epochs, verbose=1)
        # evaluate model
        _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
        print('> %.3f' % (acc * 100.0))
        # learning curves
        self.summarize_diagnostics(history, filename)

    # run the test harness for evaluating a model
    def run_test_harness_augmented(self, filename, epochs=20, dataset_home='preprocessed_dataset_vultures_vs_sharks/', verbose=1):
        # define model
        model = self.init()
        # create data generators
        train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                           width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1.0/255.0)
        # prepare iterators
        train_it = train_datagen.flow_from_directory(dataset_home+'train/',
                                                     class_mode='binary', batch_size=5, target_size=(200, 200))
        test_it = test_datagen.flow_from_directory(dataset_home+'test/',
                                                   class_mode='binary', batch_size=5, target_size=(200, 200))
        # fit model
        history = model.fit(train_it, steps_per_epoch=len(train_it),
                            validation_data=test_it, validation_steps=len(test_it), epochs=epochs, verbose=verbose)
        # evaluate model
        _, acc = model.evaluate(
            test_it, steps=len(test_it), verbose=0)
        print('> %.3f' % (acc * 100.0))
        # learning curves
        self.summarize_diagnostics(history, filename)
