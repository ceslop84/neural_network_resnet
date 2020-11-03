from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('TkAgg') # sudo apt-get install python3-tk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def filterDataset(folder, classes=None, mode='train'):
    # initialize COCO api for instance annotations
    annFile = '{}/annotations/instances_{}.json'.format(folder, mode)
    coco = COCO(annFile)

    images = []
    if classes!=None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given categories
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)

    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)

    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])

    random.shuffle(unique_images)
    dataset_size = len(unique_images)

    return unique_images, dataset_size, coco

def augmentationsGenerator(gen, augGeneratorArgs, seed=None):
    # Initialize the image data generator with args provided
    image_gen = ImageDataGenerator(**augGeneratorArgs)

    # Remove the brightness argument for the mask. Spatial arguments similar to image.
    augGeneratorArgs_mask = augGeneratorArgs.copy()
    _ = augGeneratorArgs_mask.pop('brightness_range', None)
    # Initialize the mask data generator with modified args
    mask_gen = ImageDataGenerator(**augGeneratorArgs_mask)

    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))

    for img, mask in gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation of the images
        # will end up different from the augmentation of the masks
        g_x = image_gen.flow(255*img,
                             batch_size = img.shape[0],
                             seed = seed,
                             shuffle=True)
        g_y = mask_gen.flow(mask,
                             batch_size = mask.shape[0],
                             seed = seed,
                             shuffle=True)

        img_aug = next(g_x)/255.0
        mask_aug = next(g_y)

        yield img_aug, mask_aug

def visualizeGenerator(gen):
    # Iterate the generator to get image and mask batches
    img, mask = next(gen)

    fig = plt.figure(figsize=(20, 10))
    outerGrid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)

    for i in range(2):
        innerGrid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outerGrid[i], wspace=0.05, hspace=0.05)

        for j in range(4):
            ax = plt.Subplot(fig, innerGrid[j])
            if(i==1):
                ax.imshow(img[j])
            else:
                ax.imshow(mask[j][:,:,0])

            ax.axis('off')
            fig.add_subplot(ax)
    plt.show()

def dataGeneratorCoco(images, classes, coco, folder,
                      input_image_size=(224,224), batch_size=4, mode='train', mask_type='binary'):

    img_folder = '{}/{}'.format(folder, mode)
    dataset_size = len(images)
    catIds = coco.getCatIds(catNms=classes)

    c = 0
    while(True):
        img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
        mask = np.zeros((batch_size, input_image_size[0], input_image_size[1], 1)).astype('float')

        for i in range(c, c+batch_size): #initially from 0 to batch_size, when c = 0
            imageObj = images[i]

            ### Retrieve Image ###
            train_img = getImage(imageObj, img_folder, input_image_size)

            ### Create Mask ###
            if mask_type=="binary":
                train_mask = getBinaryMask(imageObj, coco, catIds, input_image_size)

            elif mask_type=="normal":
                train_mask = getNormalMask(imageObj, classes, coco, catIds, input_image_size)

            # Add to respective batch sized arrays
            img[i-c] = train_img
            mask[i-c] = train_mask

        c+=batch_size
        if(c + batch_size >= dataset_size):
            c=0
            random.shuffle(images)
        yield img, mask

def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return None

def getImage(imageObj, img_folder, input_image_size):
    # Read and normalize an image
    train_img = io.imread(img_folder + '/' + imageObj['file_name'])/255.0
    # Resize
    train_img = cv2.resize(train_img, input_image_size)
    if (len(train_img.shape)==3 and train_img.shape[2]==3): # If it is a RGB 3 channel image
        return train_img
    else: # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,)*3, axis=-1)
        return stacked_img

def getNormalMask(imageObj, classes, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    cats = coco.loadCats(catIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        className = getClassName(anns[a]['category_id'], cats)
        pixel_value = classes.index(className)+1
        new_mask = cv2.resize(coco.annToMask(anns[a])*pixel_value, input_image_size)
        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask

def getBinaryMask(imageObj, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        new_mask = cv2.resize(coco.annToMask(anns[a]), input_image_size)

        #Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0

        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask

augGeneratorArgs = dict(featurewise_center = False,
                        samplewise_center = False,
                        rotation_range = 5,
                        width_shift_range = 0.01,
                        height_shift_range = 0.01,
                        brightness_range = (0.8,1.2),
                        shear_range = 0.01,
                        zoom_range = [1, 1.25],
                        horizontal_flip = True,
                        vertical_flip = False,
                        fill_mode = 'reflect',
                        data_format = 'channels_last')
folder = './software/mscoco_data'
classes = ['laptop', 'tv', 'cell phone']
batch_size_gen = 4
input_image_size = (224,224)
mask_type = 'binary'
epochs = 50
batch_size = 64
opt = "sgd"
lossFn = "categorical_crossentropy"

images_val, dataset_size_val, coco_val = filterDataset(folder, classes, "val2017")
val_gen = dataGeneratorCoco(images_val, classes, coco_val, folder,
                            input_image_size, batch_size_gen, "val2017", mask_type)
val_gen_aug = augmentationsGenerator(val_gen, augGeneratorArgs)
# visualizeGenerator(val_gen_aug)

images_train, dataset_size_train, coco_train = filterDataset(folder, classes, "train2017")
train_gen = dataGeneratorCoco(images_train, classes, coco_train, folder,
                            input_image_size, batch_size_gen, "train2017", mask_type)
train_gen_aug = augmentationsGenerator(train_gen, augGeneratorArgs)
#visualizeGenerator(train_gen_aug)

modelo = keras.Sequential([keras.Input(shape=(3,)),
                           keras.layers.Dense(10, activation='tanh'),
                           keras.layers.Dense(20, activation='relu'),
                           keras.layers.Dense(10, activation='linear'),
                           keras.layers.Dense(1, activation='sigmoid')])

modelo.compile(loss = lossFn, optimizer = opt, metrics=['accuracy'])

history = modelo.fit(x = train_gen_aug,
                validation_data = val_gen_aug,
                epochs = epochs,
                batch_size = batch_size,
                verbose = True)

print("fim")
