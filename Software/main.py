from math import sqrt
from datetime import datetime
import pandas as pd
import numpy as np
from tensorflow.keras import losses
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, Flatten, Dropout, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomFlip, RandomRotation, RandomContrast
from tensorflow.data.experimental import AUTOTUNE
from tensorflow_datasets import load
import matplotlib
matplotlib.use('TkAgg') # sudo apt-get install python3-tk
import matplotlib.pyplot as plt


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

def identity_block(X, kernel_size, filters):

    # Salvando valor de entrada...
    X_shortcut = X

    # Primeira camada.
    X = Conv2D(filters = filters,
               kernel_size = (1, 1),
               strides = (1,1),
               padding = 'same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Segunda camada.
    X = Conv2D(filters = filters,
               kernel_size = kernel_size,
               strides = (1,1),
               padding = 'same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(X)
    X = BatchNormalization()(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, kernel_size, strides, filters, head=False):

    # Salvando a entrada...
    X_input = X

    # Primeira camada.
    X = Conv2D(filters=filters,
               kernel_size=(1, 1),
               strides=strides,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Segunda camada.
    X = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=(1, 1),
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(X)
    X = BatchNormalization()(X)

    if not head:
        # Concatenando com a entrada original...
        X_input = Conv2D(filters=filters,
                kernel_size=(1,1),
                strides=strides,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(X_input)

    # Adicionando as camadas.
    X = Add()([X, X_input])
    X = Activation('relu')(X)

    return X

def load_model(model_name, num_labels, keep_prob, input_shape, augmented, dropout):
    """Criando a arquitetura da rede, cfe nome."""
    # Criando o modelo sequencial.

    inputs = Input(shape=input_shape)
    x = Rescaling(1./255)(inputs)

    if augmented:
        x = RandomFlip("horizontal_and_vertical")(x)
        x = RandomRotation(0.2)(x)
        x = RandomContrast(0.2)(x)

    x = Conv2D(filters=16,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if model_name == "Flat8":
        x = convolutional_block(x, filters=16, kernel_size=(3,3), strides=(1,1), head=True)
        x = convolutional_block(x, filters=32, kernel_size=(3,3), strides=(2,2))
        x = convolutional_block(x, filters=64, kernel_size=(3,3), strides=(2,2))

    elif model_name == "Flat14":
        x = convolutional_block(x, filters=16, kernel_size=(3,3), strides=(1,1), head=True)
        x = convolutional_block(x, filters=16, kernel_size=(3,3), strides=(1,1))
        x = convolutional_block(x, filters=16, kernel_size=(3,3), strides=(1,1))
        x = convolutional_block(x, filters=32, kernel_size=(3,3), strides=(2,2))
        x = convolutional_block(x, filters=32, kernel_size=(3,3), strides=(1,1))
        x = convolutional_block(x, filters=32, kernel_size=(3,3), strides=(1,1))

    elif model_name == "Flat20":
        x = convolutional_block(x, filters=16, kernel_size=(3,3), strides=(1,1), head=True)
        x = convolutional_block(x, filters=16, kernel_size=(3,3), strides=(1,1))
        x = convolutional_block(x, filters=16, kernel_size=(3,3), strides=(1,1))
        x = convolutional_block(x, filters=32, kernel_size=(3,3), strides=(2,2))
        x = convolutional_block(x, filters=32, kernel_size=(3,3), strides=(1,1))
        x = convolutional_block(x, filters=32, kernel_size=(3,3), strides=(1,1))
        x = convolutional_block(x, filters=64, kernel_size=(3,3), strides=(2,2))
        x = convolutional_block(x, filters=64, kernel_size=(3,3), strides=(1,1))
        x = convolutional_block(x, filters=64, kernel_size=(3,3), strides=(1,1))

    elif model_name == "ResNet8":
        x = convolutional_block(x, filters=16, kernel_size=(3,3), strides=(1,1), head=True)
        x = identity_block(x, filters=16, kernel_size=(3,3))
        x = identity_block(x, filters=16, kernel_size=(3,3))

    elif model_name == "ResNet14":
        x = convolutional_block(x, filters=16, kernel_size=(3,3), strides=(1,1), head=True)
        x = identity_block(x, filters=16, kernel_size=(3,3))
        x = identity_block(x, filters=16, kernel_size=(3,3))
        x = convolutional_block(x, filters=32, kernel_size=(3,3), strides=(2,2))
        x = identity_block(x, filters=32, kernel_size=(3,3))
        x = identity_block(x, filters=32, kernel_size=(3,3))

    elif model_name == "ResNet20":
        x = convolutional_block(x, filters=16, kernel_size=(3,3), strides=(1,1), head=True)
        x = identity_block(x, filters=16, kernel_size=(3,3))
        x = identity_block(x, filters=16, kernel_size=(3,3))
        x = convolutional_block(x, filters=32, kernel_size=(3,3), strides=(2,2))
        x = identity_block(x, filters=32, kernel_size=(3,3))
        x = identity_block(x, filters=32, kernel_size=(3,3))
        x = convolutional_block(x, filters=64, kernel_size=(3,3), strides=(2,2))
        x = identity_block(x, filters=64, kernel_size=(3,3))
        x = identity_block(x, filters=64, kernel_size=(3,3))

    else:
        # Se chegou até aqui, não retornou nenhum modelo pois não reconheceu o nome.
        raise Exception("Nome do modelo desconhecido/não suportado.")

    x = Flatten()(x)

    if dropout:
        x = Dropout(keep_prob)(x)
    outputs = Dense(num_labels,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def run(epochs, batch_size, optimizer, loss, metrics,
        folder, input_shape, keep_prob, train_size, val_size, test_size,
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
    checkpoint = ModelCheckpoint(filepath=folder,
                                monitor='val_acc',
                                save_best_only=True)
    callbacks.append(checkpoint)
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
EPOCHS = 50
BATCH_SIZE = 32
FOLDER = "localdata/"
INPUT_SHAPE = (32,32,3)
KEEP_PROB = 0.7
TRAIN_SIZE = 40000
VAL_SIZE = 10000
TEST_SIZE = 10000
OPTIMIZER = Adam(learning_rate=lr_schedule(0))
LOSS = losses.SparseCategoricalCrossentropy(from_logits=True)
METRICS = ['accuracy']

MODEL_NAME = ["Flat8", "Flat14", "Flat20", "ResNet8", "ResNet14", "ResNet20"]
DROPOUT = [True, False]
AUGMENTED = [True, False]
LEARN_RATE = [True, False]

for m in MODEL_NAME:
    for d in DROPOUT:
        for a in AUGMENTED:
            for l in LEARN_RATE:
                history, test_loss, test_acc = run(EPOCHS, BATCH_SIZE, OPTIMIZER, LOSS, METRICS,
                                                   FOLDER, INPUT_SHAPE, KEEP_PROB,
                                                   TRAIN_SIZE, VAL_SIZE, TEST_SIZE,
                                                   m, d, a, l)
                save_data(m, d, a, l, history, test_loss, test_acc)
