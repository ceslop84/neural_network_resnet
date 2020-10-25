import os
from math import sqrt
from datetime import datetime
import pandas as pd
import numpy as np
from tensorflow.keras import losses, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, Flatten, Dropout, Add, MaxPooling2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomFlip, RandomRotation, RandomContrast
from tensorflow.data.experimental import AUTOTUNE
from tensorflow_datasets import load


def resnet(input_shape, depth, num_classes, augmented, dropout, keep_prob):
    
    num_filters = 16
    num_res_blocks = int((depth - 2)/6)

    inputs = Input(shape=input_shape)
    rescale = Sequential([Rescaling(1./255)])
    x = rescale(inputs)

    if augmented:
        data_augmentation = Sequential([RandomFlip("horizontal_and_vertical"),
                                        RandomRotation(0.2),
                                        RandomContrast(0.2)])
        x = data_augmentation(x)

    x = net_layer(inputs=x)

    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = net_layer(inputs=x,
                          num_filters=num_filters,
                          strides=strides)
            y = net_layer(inputs=y,
                          num_filters=num_filters,
                          activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = net_layer(inputs=x,
                              num_filters=num_filters,
                              kernel_size=1,
                              strides=strides,
                              activation=None,
                              batch_normalization=False)
            x = Add()([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    if dropout:
        x = Dropout(keep_prob)(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def flatnet(input_shape, depth, num_classes, augmented, dropout, keep_prob):
    
    num_filters = 16
    num_res_blocks = int((depth - 2)/6)

    inputs = Input(shape=input_shape)
    rescale = Sequential([Rescaling(1./255)])
    x = rescale(inputs)

    if augmented:
        data_augmentation = Sequential([RandomFlip("horizontal_and_vertical"),
                                        RandomRotation(0.2),
                                        RandomContrast(0.2)])
        x = data_augmentation(x)

    x = net_layer(inputs=x)

    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            x = net_layer(inputs=x,
                          num_filters=num_filters,
                          strides=strides)
            x = net_layer(inputs=x,
                          num_filters=num_filters,
                          activation=None)
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    if dropout:
        x = Dropout(keep_prob)(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def save_data(model_name, dropout, augmented, learn_rate, history, test_loss, test_acc):
    #Geração do nome do arquivo.
    name = model_name + "_D" + str(int(dropout)) + "_A" + str(int(augmented)) + "_L" + str(int(learn_rate))
    # Geração do arquivo de saída.
    hist = history.history
    hist["test_loss"] = test_loss
    hist["test_acc"] = test_acc
    saida_df = pd.DataFrame(hist)
    saida_df.columns = ["loss", "acc", "val_loss", "val_acc", "lr", "test_loss", "test_acc"]
    # Registro da hora de início para a geração dos arquivos de saída em pasta específica.
    timestamp = str(datetime.today().strftime('%Y%m%d_%H%M%S'))
    saida_df.to_csv(timestamp + "_" + name + ".csv", index=False)

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
    if epoch > 90:
        lr *= 0.5e-3
    elif epoch > 80:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def configure_for_performance(ds,
                             batch_size,
                             shuffle_bs=1000,
                             prefetch_bs=AUTOTUNE):
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

def net_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    
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

def load_model(model_name, num_labels, keep_prob, input_shape, augmented, dropout):
    """Criando a arquitetura da rede, cfe nome."""
    # Criando o modelo sequencial.

    if model_name == "Flat8":
        model = flatnet(input_shape=input_shape, depth=8, num_classes=num_labels, augmented=augmented, dropout=dropout, keep_prob=keep_prob)

    elif model_name == "Flat14":
        model = flatnet(input_shape=input_shape, depth=14, num_classes=num_labels, augmented=augmented, dropout=dropout, keep_prob=keep_prob)

    elif model_name == "Flat20":
        model = flatnet(input_shape=input_shape, depth=20, num_classes=num_labels, augmented=augmented, dropout=dropout, keep_prob=keep_prob)

    elif model_name == "Flat26":
        model = flatnet(input_shape=input_shape, depth=26, num_classes=num_labels, augmented=augmented, dropout=dropout, keep_prob=keep_prob)

    elif model_name == "Flat32":
        model = flatnet(input_shape=input_shape, depth=32, num_classes=num_labels, augmented=augmented, dropout=dropout, keep_prob=keep_prob)

    elif model_name == "ResNet8":
        model = resnet(input_shape=input_shape, depth=8, num_classes=num_labels, augmented=augmented, dropout=dropout, keep_prob=keep_prob)

    elif model_name == "ResNet14":
        model = resnet(input_shape=input_shape, depth=14, num_classes=num_labels, augmented=augmented, dropout=dropout, keep_prob=keep_prob)

    elif model_name == "ResNet20":
        model = resnet(input_shape=input_shape, depth=20, num_classes=num_labels, augmented=augmented, dropout=dropout, keep_prob=keep_prob)

    elif model_name == "ResNet26":
        model = resnet(input_shape=input_shape, depth=26, num_classes=num_labels, augmented=augmented, dropout=dropout, keep_prob=keep_prob)

    elif model_name == "ResNet32":
        model = resnet(input_shape=input_shape, depth=32, num_classes=num_labels, augmented=augmented, dropout=dropout, keep_prob=keep_prob)


    else:
        # Se chegou até aqui, não retornou nenhum modelo pois não reconheceu o nome.
        raise Exception("Nome do modelo desconhecido/não suportado.")

    model.summary()
    return model

def run(epochs, batch_size, optimizer, loss, metrics,
        input_shape, keep_prob, train_size, val_size, test_size,
        model_name, dropout, augmented, learn_rate):

    # Ler os dados do modelo.
    train_val_dataset, test_dataset, data_info = load_datasets()

    # Separar os dados de treinamento e validação.
    train_dataset, val_dataset = split_train_val_dataset(train_val_dataset, data_info)

    # Processa as informaões dos nomes e quantidades de classes.
    labels, num_labels, get_label = process_labels(data_info)

    # Configura os dados para melhor desempenho.
    train_dataset = configure_for_performance(train_dataset, batch_size, shuffle_bs=train_size)
    val_dataset = configure_for_performance(val_dataset, batch_size, shuffle_bs=val_size)
    test_dataset = configure_for_performance(test_dataset, batch_size, shuffle_bs=test_size)

    # Ler o modelo.
    model = load_model(model_name, num_labels, keep_prob, input_shape, augmented, dropout)

    # Compilar modelo.
    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)

   # Prepare callbacks for model saving and for learning rate adjustment.
    callbacks = list()
    if learn_rate:
        lr_scheduler = LearningRateScheduler(lr_schedule)
        callbacks.append(lr_scheduler)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    callbacks.append(lr_reducer)

    # Treinar modelo com dados de treinamento e validação.
    history = model.fit(train_dataset,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=val_dataset,
                        shuffle=True,
                        callbacks=callbacks)

    # Avaliar modelo contra os dados de testes.
    test_loss, test_acc = model.evaluate(test_dataset)

    return history, test_loss, test_acc

# Parâmetros de entrada.
EPOCHS = 100
BATCH_SIZE = 32
INPUT_SHAPE = (32,32,3)
KEEP_PROB = 0.7
TRAIN_SIZE = 40000
VAL_SIZE = 10000
TEST_SIZE = 10000
OPTIMIZER = Adam(learning_rate=lr_schedule(0))
LOSS = losses.SparseCategoricalCrossentropy()
METRICS = ['accuracy']

MODEL_NAME = ["ResNet32", "Flat32", "ResNet8", "Flat8", "ResNet20", "Flat20"]
DROPOUT = [True, False]
AUGMENTED = [True, False]
LEARN_RATE = [True, False]

for m in MODEL_NAME:
    for d in DROPOUT:
        for a in AUGMENTED:
            for l in LEARN_RATE:
                history, test_loss, test_acc = run(EPOCHS, BATCH_SIZE, OPTIMIZER, LOSS, METRICS,
                                                   INPUT_SHAPE, KEEP_PROB,
                                                   TRAIN_SIZE, VAL_SIZE, TEST_SIZE,
                                                   m, d, a, l)
                save_data(m, d, a, l, history, test_loss, test_acc)
