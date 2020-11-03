#https://www.tensorflow.org/tutorials/images/cnn
from math import sqrt
from tensorflow_datasets import load
from tensorflow.data import Dataset, experimental
from tensorflow.keras import layers, models, losses, Input
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
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

def split_train_val_dataset(dataset, num_labels):
    train_list = list()
    val_list = list()
    for i in range(num_labels):
        label_ds = dataset.filter(lambda img, lbl: lbl == (i+1))
        train_ds = label_ds.shard(num_shards=5, index=0)
        for i in range(1,4):
            train_ds.concatenate(label_ds.shard(num_shards=5, index=i))
        val_ds = label_ds.shard(num_shards=5, index=4)
        train_list.append(train_ds)
        val_list.append(val_ds)
        del label_ds, train_ds, val_ds

    train_ds = train_list[0]
    for i in range(1,len(train_list)):
        train_ds.concatenate(train_list[i])

    val_ds = val_list[0]
    for i in range(1,len(val_list)):
        val_ds.concatenate(val_list[i])

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

    train_dataset, val_dataset = split_train_val_dataset(train_val_dataset,
                                                         data_info.features['label'].num_classes)
    return train_dataset, val_dataset, test_dataset, data_info

def process_labels(data_info):
    labels = data_info.features['label'].names
    num_labels = data_info.features['label'].num_classes
    get_label = data_info.features['label'].int2str
    return labels, num_labels, get_label

EPOCHS = 10
BATCH_SIZE = 128
OPTIMIZER = 'RMSProp'
LOSS = losses.SparseCategoricalCrossentropy(from_logits=True)
METRICS = ['accuracy']
AUTOTUNE = experimental.AUTOTUNE

train_dataset, val_dataset, test_dataset, data_info = load_datasets()
labels, num_labels, get_label = process_labels(data_info)

train_dataset = configure_for_performance(train_dataset, BATCH_SIZE, prefetch_bs=AUTOTUNE)
val_dataset = configure_for_performance(val_dataset, BATCH_SIZE, prefetch_bs=AUTOTUNE)
test_dataset = configure_for_performance(test_dataset, BATCH_SIZE, prefetch_bs=AUTOTUNE)

model = models.Sequential([Input(shape=(32,32,3)),
                           # Data augmentation.
                           layers.experimental.preprocessing.RandomContrast(0.2),
                           layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                           layers.experimental.preprocessing.RandomRotation(0.2),
                           # Rescaling data.
                           layers.experimental.preprocessing.Rescaling(scale=1./255),
                           # Neural Network itself.
                           layers.Conv2D(32, (3, 3), activation='relu'),
                           layers.MaxPooling2D((2, 2)),
                           layers.Conv2D(64, (3, 3), activation='relu'),
                           layers.MaxPooling2D((2, 2)),
                           layers.Conv2D(64, (3, 3), activation='relu'),
                           layers.Flatten(),
                           layers.Dense(64, activation='relu'),
                           layers.Dense(num_labels)])

model.compile(optimizer=OPTIMIZER,
              loss=LOSS,
              metrics=METRICS)

history = model.fit(train_dataset,
                    batch_size = BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=val_dataset)

plot_graph(history)

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)

print(test_acc)
