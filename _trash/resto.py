    if model_name == "cnn_aug1":
        # Data augmentation.
        model.add(layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"))
        # Rescaling data.
        model.add(layers.experimental.preprocessing.Rescaling(scale=1./255))

        # Model...
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())

        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(num_labels))

        return model

    if model_name == "cnn_aug2":
        # Data augmentation.
        model.add(layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"))
        model.add(layers.experimental.preprocessing.RandomRotation(0.2))
        # Rescaling data.
        model.add(layers.experimental.preprocessing.Rescaling(scale=1./255))

        # Model...
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())

        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(num_labels))

        return model

    if model_name == "cnn_aug3":
        # Data augmentation.
        model.add(layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"))
        model.add(layers.experimental.preprocessing.RandomRotation(0.2))
        model.add(layers.experimental.preprocessing.RandomContrast(0.2))
        # Rescaling data.
        model.add(layers.experimental.preprocessing.Rescaling(scale=1./255))

        # Model...
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())

        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(num_labels))

        return model
