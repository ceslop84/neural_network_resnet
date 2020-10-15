from math import sqrt
#from tensorflow.data import experimental
from tensorflow.keras import layers, models, losses
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, AveragePooling2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow_datasets import load
import matplotlib
matplotlib.use('TkAgg') # sudo apt-get install python3-tk
import matplotlib.pyplot as plt
import numpy as np


def visualize(dataset, labels, size):
    figplot = int(sqrt(size))
    figsize = figplot * 2
    plt.figure(figsize=(figsize, figsize))
    i = 0
    for img in dataset.take(size):
        plt.subplot(figplot, figplot,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img["image"], cmap=plt.cm.binary)
        plt.xlabel(labels[img["label"]])
        i += 1
    plt.show()

def plot_graph(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

def split_train_val_dataset(dataset, data_info):
    num_labels = data_info.features['label'].num_classes
    for i in range(num_labels):
        label_ds = dataset.filter(lambda img, lbl: lbl == i)
        if i == 0 :
            train_ds = label_ds.shard(num_shards=5, index=0)
            val_ds = label_ds.shard(num_shards=5, index=4)
        else:
            train_ds = train_ds.concatenate(label_ds.shard(num_shards=5, index=0))
            val_ds = val_ds.concatenate(label_ds.shard(num_shards=5, index=4))
        for j in range(1,4):
            train_ds = train_ds.concatenate(label_ds.shard(num_shards=5, index=j))
    return train_ds, val_ds

def configure_for_performance(ds,
                             batch_size,
                             shuffle_bs=1000,
                             prefetch_bs=experimental.AUTOTUNE):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_bs)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=prefetch_bs)
    return ds

def load_datasets():
    (train_val_dataset, test_dataset), data_info = load("cifar10",
                                                        split=["train", "test"],
                                                        as_supervised=True,
                                                        with_info=True)

    return train_val_dataset, test_dataset, data_info

def process_labels(data_info):
    labels = data_info.features['label'].names
    num_labels = data_info.features['label'].num_classes
    get_label = data_info.features['label'].int2str
    return labels, num_labels, get_label

def count_elements(dataset):
    count = [i for i,_ in enumerate(dataset)][-1] + 1
    return count

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet_v2(input_shape, depth, num_classes):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def load_model(model_name, num_labels, keep_prob, input_shape):
    # Criando a arquitetura da rede, cfe nome.
    if model_name == "cnn8":
        # Criando o modelo sequencial.
        model = models.Sequential()
        model.add(Input(shape=input_shape))
        # Rescaling data.
        model.add(experimental.preprocessing.Rescaling(scale=1./255))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())

        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_labels))

        return model

    if model_name == "cnn14":
        # Criando o modelo sequencial.
        model = models.Sequential()
        model.add(Input(shape=input_shape))
        # Rescaling data.
        model.add(experimental.preprocessing.Rescaling(scale=1./255))

        # Neural Network itself.
        model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
        model.add(MaxPooling2D((2, 2), (2,2), padding="same"))
        model.add(BatchNormalization())

        model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
        model.add(MaxPooling2D((2, 2), (2,2), padding="same"))
        model.add(BatchNormalization())

        model.add(Conv2D(256, (3, 3), activation='relu', padding="same"))
        model.add(MaxPooling2D((2, 2), (2,2), padding="same"))
        model.add(BatchNormalization())

        model.add(Conv2D(512, (3, 3), activation='relu', padding="same"))
        model.add(MaxPooling2D((2, 2), (2,2), padding="same"))
        model.add(BatchNormalization())

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(keep_prob))
        model.add(BatchNormalization())

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(keep_prob))
        model.add(BatchNormalization())

        model.add(Dense(512, activation='relu'))
        model.add(Dropout(keep_prob))
        model.add(BatchNormalization())

        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(keep_prob))
        model.add(BatchNormalization())

        model.add(Dense(num_labels))

        return model

    if model_name == "ResNet20v1":
        model = resnet_v1(input_shape=input_shape, depth=20, num_classes=num_labels)
        return model

    if model_name == "ResNet20v2":
        model = resnet_v2(input_shape=input_shape, depth=20, num_classes=num_labels)
        return model

    if model_name == "ResNet32v1":
        model = resnet_v1(input_shape=input_shape, depth=32, num_classes=num_labels)
        return model

    if model_name == "ResNet56v2":
        model = resnet_v2(input_shape=input_shape, depth=56, num_classes=num_labels)
        return model

    if model_name == "ResNet110v2":
        model = resnet_v2(input_shape=input_shape, depth=110, num_classes=num_labels)
        return model

    # Se chegou até aqui, não retornou nenhum modelo pois não reconheceu o nome.
    raise Exception("Nome do modelo desconhecido/não suportado.")

def run(epochs, batch_size, optimizer, loss, metrics, model_name, folder, input_shape):
    # Ler os dados do modelo.
    train_val_dataset, test_dataset, data_info = load_datasets()
    # Separar os dados de treinamento e validação.
    train_dataset, val_dataset = split_train_val_dataset(train_val_dataset, data_info)
    # Processa as informaões dos nomes e quantidades de classes.
    labels, num_labels, get_label = process_labels(data_info)
    # Configura os dados para melhor desempenho.
    train_dataset = configure_for_performance(train_dataset, batch_size, shuffle_bs=40000)
    val_dataset = configure_for_performance(val_dataset, batch_size, shuffle_bs=10000)
    test_dataset = configure_for_performance(test_dataset, batch_size, shuffle_bs=10000)
    # Ler o modelo.
    model = load_model(model_name, num_labels, 0.7, input_shape)
    # Compilar modelo.
    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)
    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=folder,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown=0,
                                patience=5,
                                min_lr=0.5e-6)
    callbacks = [checkpoint, lr_reducer, lr_scheduler]
    # Treinar modelo com dados de treinamento e validação.
    history = model.fit(train_dataset,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=val_dataset,
                        shuffle=True,
                        callbacks=callbacks)
    # Visualizar resultado do treinamento.
    plot_graph(history)
    # Avaliar modelo contra os dados de testes.
    test_loss, test_acc = model.evaluate(test_dataset)
    # Visualizar resultado do teste.
    print(f"Loss: {test_loss}    Accuracy: {test_acc}")

# Parâmetros de entrada.
EPOCHS = 12
BATCH_SIZE = 32
OPTIMIZER = Adam(learning_rate=lr_schedule(0))
LOSS = losses.SparseCategoricalCrossentropy(from_logits=True)
METRICS = ['accuracy']
MODEL_NAME = "cnn14"
FOLDER = "localdata/"
INPUT_SHAPE = (32,32,3)

run(EPOCHS, BATCH_SIZE, OPTIMIZER, LOSS, METRICS, MODEL_NAME, FOLDER, INPUT_SHAPE)
