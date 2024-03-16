import numpy as np
import glob
import cv2
import os
import pandas as pd
import shutil
import random
import warnings
from sklearn import preprocessing
from sklearn.decomposition import PCA
import itertools


# This function is used to receive dataset paths, store wanted images in training and testing paths, and the size of the images to be resized to.
# The function returns the preprocessed data to be used in an ML model of our choice.
def preprocess_data(dataset_path, train_path, test_path, n, ratio, size, pca_comps):
    """
    Preprocesses the data to be used in the model.
    :param dataset_path: The path to the dataset.
    :param train_path: The path to the training images.
    :param test_path: The path to the testing images.
    :param n: The number of images to sample from the dataset.
    :param ratio: The ratio of the sampled data to be used for training.
    :param size: The row/column size to which the images will be resized.
    :param pca_comps: The number of components to reduce the dataset to using PCA.
    :return: The preprocessed data.
    """

    sample_images(dataset_path, train_path, test_path, n, ratio)  # Sample the images from the dataset to the training and testing paths

    x_train, y_train, x_test, y_test = resize_and_split(train_path, test_path, size)  # Resize the images and split the data into training and testing datasets

    y_train_encoded, y_test_encoded, le = label_data(y_train, y_test)  # Convert the labels to numbers for the model to be able to process them

    x_train = minmax_normalize(x_train)  # Normalize the data to be between 0 and 1 using the min-max normalization
    x_test  = minmax_normalize(x_test)   # Normalize the data to be between 0 and 1 using the min-max normalization

    x_train_features = feature_extraction(x_train, pca_comps)                         # Extract features from the images and align them in a dataframe
    x_train_features = np.expand_dims(x_train_features, axis=0)                       # Expand the dimensions of the training dataset to be used in the model
    x_train_features = np.reshape(x_train_features, (x_train.shape[0], -1))  # Reshape the training dataset to be used to train model

    x_test_features = feature_extraction(x_test, pca_comps)                                    # Extract features from the images and align them in a dataframe
    x_test_features = np.expand_dims(x_test_features, axis=0)                       # Expand the dimensions of the testing dataset
    x_test_features = np.reshape(x_test_features, (x_test.shape[0], -1))   # Reshape the testing dataset to be used in the trained model

    return x_train_features, y_train_encoded, x_test_features, y_test_encoded, le


def sample_images(data_path, train_path, test_path, n, ratio=0.8):
    """
    This method samples the images from the dataset path to the training path and testing path.
    :param data_path: The path to the data.
    :param train_path: The path to which the training images will be copied.
    :param test_path: The path to which the testing images will be copied.
    :param n: The number of images aimed to be sampled.
    :param ratio: The wanted ratio from the sampled data to be used for training. The rest will be used for testing.
    :return: None
    """

    if ratio >= 1 or ratio <= 0:
        raise ValueError("The ratio should be between 0 and 1 (non-inclusive)")

    if ratio < 0.5:
        warnings.warn("The ratio is less than 0.5, not advised for good training")

    artists = os.listdir(data_path)  # list of artists

    for artist in artists:
        artist_path = os.path.join(data_path, artist)  # path to the artist
        images = os.listdir(artist_path)  # list of images
        random.shuffle(images)  # shuffle the images inside the images list

        # Adjust the number of images based on availability
        n_train = min(int(n * ratio), int(len(images) * ratio))  # 80% of the images
        n_test = min(n - n_train, len(images) - n_train)  # 20% of the images

        # Partition the images to training and testing
        train_images = images[:n_train]
        test_images = images[n_train:n_train + n_test]

        # Create the directories for training and testing
        artist_train_path = os.path.join(train_path, artist)
        artist_test_path = os.path.join(test_path, artist)
        os.makedirs(artist_train_path, exist_ok=True)
        os.makedirs(artist_test_path, exist_ok=True)

        # Create the directories
        for image in train_images:
            image_path = os.path.join(artist_path, image)
            shutil.copy(image_path, os.path.join(artist_train_path, image))

        for image in test_images:
            image_path = os.path.join(artist_path, image)
            shutil.copy(image_path, os.path.join(artist_test_path, image))


def clear_files(directory):
    """
    Clears all files within the subdirectories of the given directory.
    :param directory: The directory to clear its subcategories' files.
    """

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)


def resize_and_split(train_path, test_path, size=128):
    """
    Resizes the images to the given size and splits the data into training and testing datasets.
    :param train_path: The path from which the training images will be taken.
    :param test_path: The path from which the testing images will be taken.
    :param size: The row/column size to which the images will be resized.
    :return: Lists of the training and testing images and their corresponding labels.
    """
    train_images = []
    train_labels = []

    # The following loop reads the images from the training path, resizes them, and adds them to the list of images.
    # It also adds the corresponding label for each image to the list of labels.
    for directory_path in glob.glob(train_path+r'/*'):
        label = directory_path.split("\\")[-1]  # extracts the artist name from the directory path.
        # print(label)
        for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Read the image in color (BGR format)
            img = cv2.resize(img, (size, size))     # Resize the image to the given size, using weighted average values for interpolation.
            train_images.append(img)    # Add the processed image to the list of images
            train_labels.append(label)  # Add the label to the list of labels, matching the image in the same index in the images list.

    # Convert the lists to numpy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    ##################

    test_images = []
    test_labels = []

    # The following loop reads the images from the training path, resizes them, and adds them to the list of images.
    # It also adds the corresponding label for each image to the list of labels.
    for directory_path in glob.glob(test_path+r'/*'):
        label = directory_path.split("\\")[-1]  # extracts the artist name from the directory path.
        # print(label)
        for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Read the image in color (BGR format)
            img = cv2.resize(img, (size, size))     # Resize the image to the given size, using weighted average values for interpolation.
            test_images.append(img)    # Add the processed image to the list of images
            test_labels.append(label)  # Add the label to the list of labels, matching the image in the same index in the images list.

    # Convert the lists to numpy arrays
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    return train_images, train_labels, test_images, test_labels


def label_data(train_labels, test_labels):
    """
    Converts the labels to numbers for the model to be able to process them, utilizing sklearn LabelEncoder.
    :param train_labels: The labels of the images in the training dataset.
    :param test_labels: The labels of the images in the testing dataset.
    :return: The encoded labels for both the training and testing datasets.
    """
    # Converting the labels to numbers for the model to be able to process them.
    le = preprocessing.LabelEncoder()
    le.fit(test_labels)
    test_labels_encoded = le.transform(test_labels)
    le.fit(train_labels)
    train_labels_encoded = le.transform(train_labels)

    # Giving our data conventional names for easier use in the model.
    return train_labels_encoded, test_labels_encoded, le


def standard_normalize(train_images):
    """
    Normalizes the data to be between 0 and 1 using the standard normalization.
    :param train_images: The training dataset.
    :return: Normalized training and testing datasets.
    """
    images_normalized = []
    for img in train_images:
        mean = np.mean(img)
        std_dev = np.std(img)
        normalized_img = (img.astype(np.float32) - mean) / std_dev
        images_normalized.append(normalized_img)

    return np.array(images_normalized)


def minmax_normalize(images):
    """
    Normalizes the data to be between 0 and 1 using the min-max normalization.
    :param images: The training dataset.
    :return: Normalized training and testing datasets.
    """
    images_normalized = []
    for img in images:
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val == min_val:
            normalized_img = np.full_like(img, 0.5)
        else:
            normalized_img = (img.astype(np.float32) - min_val) / (max_val - min_val)
        images_normalized.append(normalized_img)

    return np.array(images_normalized)


def regular_normalize(images):
    """
    Normalizes the data to be between 0 and 1 using the normal normalization.
    :param images: The training dataset.
    :return: Normalized training and testing datasets.
    """
    images_normalized = []
    for img in images:
        normalized_img = img.astype(np.float32) / 255
        images_normalized.append(normalized_img)

    return np.array(images_normalized)


def vector_images(dataset, pca_comps):
    """
    Vectorizes the images in the dataset into 1D arrays.
    :param dataset: The dataset of images to be vectorized.
    :param pca_comps: The number of components to reduce an image using PCA.
    :return: The vectorized images.
    """
    vectorized_images = []
    for img in dataset:
        reduced_image = pca_reduction(img, pca_comps)
        vectorized_img = reduced_image.reshape(-1)
        vectorized_images.append(vectorized_img)
    return np.array(vectorized_images)


def create_gabor_filters(freq, orient, aspect, std_dev, phase_offset, kernel_size):
    """
    Creates a Gabor filter based on the given parameters.
    :param freq: The frequency of the sine component.
    :param orient: The orientation of the filter.
    :param aspect: The spatial aspect ratio of the filter.
    :param std_dev: The standard deviation of the filter.
    :param phase_offset: The phase offset of the filter.
    :param kernel_size: The kernel size of the filter (K x K).
    :return: A Gabor filters list over all possible combinations of the given parameters.
    """
    combos = list(itertools.product(freq, orient, aspect, std_dev, phase_offset, kernel_size))  # All possible combinations of the filter parameters
    filters = []

    for freq, orient, aspect, std_dev, phase_offset, kernel_size in combos:
        gabor_filter = cv2.getGaborKernel((kernel_size, kernel_size), std_dev, orient, freq, aspect, phase_offset, ktype=cv2.CV_32F)
        filters.append(gabor_filter)

    return filters


# This function takes a dataset of images and a list of Gabor filters, and applies the filters to the images.
# The result is a list of images filters by each individual filter.
def gabor_images(filter_list, dataset, pca_comps):
    """
    Applies the Gabor filters to the images in the dataset.
    :param filter_list: The list of Gabor filters to be applied to the images.
    :param dataset: The dataset of images to be filtered.
    :param pca_comps: The number of components to reduce an image using PCA.
    :return: A list of images filtered by each individual filter.
    """
    gabor_images_dict = {}
    count = 1

    # This loop applies the Gabor filters to the images in the dataset and adds the filtered images to the list of filtered images.
    for filt in filter_list:

        curr_gabor_images = []

        gabor_label = 'Gabor' + str(count)
        for image in range(dataset.shape[0]):

            input_image = dataset[image, :, :, :]  # Get the image from the dataset
            img = input_image                      # Copy the image to a new variable

            gabor_image = cv2.filter2D(img, -1, filt)
            gabor_image = pca_reduction(gabor_image, pca_comps)  # Normalize the filtered image to be between 0 and 1 (min-max normalization)
            gabor_image = gabor_image.reshape(-1)
            curr_gabor_images.append(gabor_image)

        gabor_images_dict[gabor_label] = curr_gabor_images
        count += 1

    return gabor_images_dict


def sobel_images(kernel_sizes, dataset, pca_comps):
    """
    Applies the Sobel filter to the images in the dataset.
    :param kernel_sizes: The kernel size of the Sobel filter.
    :param dataset: The dataset of images to be filtered.
    :param pca_comps: The number of components to reduce an image using PCA.
    :return: The filtered images.
    """
    sobel_images_dict = {}   # A dictionary to store the filtered images
    count = 1                 # A counter to label the Sobel images for each kernel size

    for kernel_size in kernel_sizes:

        curr_sobel_color = []              # A list to store the filtered images for the current kernel size
        curr_sobel_gray = []
        sobel_label_color = 'Sobel_Color_' + str(count)  # A label for the Sobel images for the current kernel size
        sobel_label_gray = 'Sobel_Gray_' + str(count)    # A label for the Sobel images for the current kernel size

        for img in dataset:

            channels = []  # A list to store the filtered images for each channel (BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       # Convert the image to grayscale
            blue, green, red = cv2.split(img)                  # Split the image into its BGR channels

            for channel in [blue, green, red, gray]:

                blur = cv2.GaussianBlur(channel, (kernel_size, kernel_size), 0)  # Apply Gaussian smoothing to the image
                edge_sobel_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, kernel_size)         # Apply the x-axis Sobel filter to the smoothed grayscale image
                edge_sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, kernel_size)         # Apply the y-axis Sobel filter to the smoothed grayscale image
                edge_sobel = cv2.magnitude(edge_sobel_x, edge_sobel_y)                        # Calculate the magnitude of the Sobel filter in both axes
                # edge_sobel = minmax_normalize(edge_sobel)  # Normalize the filtered image to be between 0 and 1 (min-max normalization
                channels.append(edge_sobel)                                                   # Add the filtered image to the list of filtered images

            edge_sobel_color = cv2.merge([channels[0], channels[1], channels[2]])      # Merge each color's magnitude with itself to create a 3-channel image (BGR format)
            edge_sobel_color = pca_reduction(edge_sobel_color, pca_comps)  # Reduce the image to the given number of components using PCA

            channels[3] = pca_reduction(channels[3], pca_comps)  # Reduce the image to the given number of components using PCA
            edge_sobel_gray = cv2.merge([channels[3], channels[3], channels[3]])       # Merge the gray magnitude with itself to create a 3-channel image (BGR format)

            edge_sobel_color = edge_sobel_color.reshape(-1)  # Reshape the 3-channel image to be a 1D array
            edge_sobel_gray = edge_sobel_gray.reshape(-1)    # Reshape the 3-channel image to be a 1D array

            curr_sobel_color.append(edge_sobel_color)  # Add the color filtered image to the list of filtered images
            curr_sobel_gray.append(edge_sobel_gray)    # Add the gray filtered image to the list of filtered images

        sobel_images_dict[sobel_label_color] = curr_sobel_color  # Add the list of color filtered images to the dictionary of filtered images
        sobel_images_dict[sobel_label_gray] = curr_sobel_gray    # Add the list of gray filtered images to the dictionary of filtered images
        count += 1  # Increment the counter for the next label.

    return sobel_images_dict


def pca_reduction(image, pca_components):
    """
    Reduces the image to the given number of components using PCA.
    :param image: The image to be reduced.
    :param pca_components: The number of components to reduce the image to.
    :return: The reduced image.
    """
    # If the image is colored, enters here
    if len(image.shape) == 3:
        b, g, r = cv2.split(image)  # Split the image into its BGR channels

        pca_b = PCA(n_components=pca_components)  # Create a PCA object for the red channel
        pca_g = PCA(n_components=pca_components)  # Create a PCA object for the green channel
        pca_r = PCA(n_components=pca_components)  # Create a PCA object for the blue channel

        reduced_b = pca_b.fit_transform(b)  # Fit the PCA object to the red channel and transform it
        reduced_g = pca_g.fit_transform(g)  # Fit the PCA object to the green channel and transform it
        reduced_r = pca_r.fit_transform(r)  # Fit the PCA object to the blue channel and transform it

        reduced_image = cv2.merge([reduced_b, reduced_g, reduced_r])  # Merge the reduced channels to form the reduced image
        reduced_image = reduced_image  # Reshape the reduced image to be a 1D array

        return reduced_image

    # If the image is grayscale, enters here
    else:
        pca = PCA(n_components=pca_components)
        reduced_image = pca.fit_transform(image)
        return reduced_image


# This function receives a flat image dataset, the original dataset of the images, and the number of components to reduce the dataset to.
# It returns the reduced dataset.
# This function receives a flat image dataset, the original dataset of the images, and the number of components to reduce the dataset to.
# It returns the reduced dataset.
def dataset_pca_reduction(dataset, original_dataset, pca_components):
    """
    Reduces the dataset to the given number of components using PCA.
    :param dataset: The dataset to be reduced.
    :param original_dataset: The original shape of the images in the dataset.
    :param pca_components: The number of components to reduce the dataset to.
    :return: The reduced dataset.
    """
    reduced_dataset = []
    num_images, num_rows, num_cols, num_channels = original_dataset.shape  # Get the number of images, rows, columns, and channels in the dataset

    for image in dataset:
        image = image.reshape(num_rows, num_cols, num_channels)
        b, g, r = cv2.split(image)  # Split the image into its BGR channels

        pca_b = PCA(n_components=pca_components)  # Create a PCA object for the red channel
        pca_g = PCA(n_components=pca_components)  # Create a PCA object for the green channel
        pca_r = PCA(n_components=pca_components)  # Create a PCA object for the blue channel

        reduced_b = pca_b.fit_transform(b)  # Fit the PCA object to the red channel and transform it
        reduced_g = pca_g.fit_transform(g)  # Fit the PCA object to the green channel and transform it
        reduced_r = pca_r.fit_transform(r)  # Fit the PCA object to the blue channel and transform it

        reduced_image = cv2.merge([reduced_b, reduced_g, reduced_r])  # Merge the reduced channels to form the reduced image
        reduced_image = reduced_image.reshape(-1)  # Reshape the reduced image to be a 1D array

        reduced_dataset.append(reduced_image)  # Add the reduced image to the list of reduced images

    return np.array(reduced_dataset)


# The following function is used to vectorize the images by extracting features from them, and aligning them in a dataframe.
# The input must be a 4 dimensional array. In our case, an array of colored images. Won't work with grayscale images.
def feature_extraction(dataset, pca_components):

    # The following parameters are used to create the Gabor and Sobel filters.
    f  = [0.1, 0.5]             # Represents the frequency of the sine component
    o  = [0, 90]                # Represents the orientation of the filter
    sa = [1.0]                  # Represents the spatial aspect ratio of the filter.
    sd = [0.5, 1.0]             # Represents the standard deviation of the filter
    p  = [0, np.pi/2]           # Represents the phase offset of the filter
    ks = [3, 5, 7]              # Represents the kernel size of the filter (K x K)

    filters = create_gabor_filters(f, o, sa, sd, p, ks)  # Create the Gabor filters based on the parameters above

    original_images = vector_images(dataset, pca_components)               # Reduce the images to the given number of components using PCA
    gabor_images_dict = gabor_images(filters, dataset, pca_components)     # Apply the Gabor filters to the images in the dataset
    sobel_images_dict = sobel_images(ks, dataset, pca_components)          # Apply the Sobel filter to the images in the dataset

    '''
    # Reducing the images' y-axis components using PCA
    reduced_original_images = dataset_pca_reduction(original_images, dataset, pca_components)

    reduced_gabor_images_dict = {}
    for label, images_pixels in gabor_images_dict.items():
        reduced_gabor_images_dict[label] = dataset_pca_reduction(images_pixels, dataset, pca_components)

    reduced_sobel_images_dict = {}
    for label, images_pixels in sobel_images_dict.items():
        reduced_sobel_images_dict[label] = dataset_pca_reduction(images_pixels, dataset, pca_components)
    '''

    combined_df = pd.DataFrame()
    combined_df['Original'] = np.concatenate(original_images)  # Add the original images to the dataframe

    for label, images_pixels in gabor_images_dict.items():
        combined_df[label] = np.concatenate(images_pixels)

    for label, images_pixels in sobel_images_dict.items():
        combined_df[label] = np.concatenate(images_pixels)

    return combined_df
