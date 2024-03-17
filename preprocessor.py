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
from sklearn.model_selection import train_test_split
import itertools


# This function is used to receive dataset paths, store wanted images in training and testing paths, and the size of the images to be resized to.
# The function returns the preprocessed data to be used in an ML model of our choice.
def preprocess_data(dataset_path, n, ratio, size, pca_comps_thresh):
    """
    Preprocesses the data to be used in the model.
    :param dataset_path: The path to the dataset.
    :param n: The number of images to sample from the dataset.
    :param ratio: The ratio of the sampled data to be used for training.
    :param size: The row/column size to which the images will be resized.
    :param pca_comps_thresh: The threshold for the number of components to reduce the dataset to using PCA.
    :return: The preprocessed data.
    """

    '''
    x_train, y_train, x_test, y_test = resize_and_split(train_path, test_path, size)  # Resize the images and split the data into training and testing datasets

    y_train_encoded, y_test_encoded, le = label_data(y_train, y_test)  # Convert the labels to numbers for the model to be able to process them

    x_train = minmax_normalize(x_train)  # Normalize the data to be between 0 and 1 using the min-max normalization
    x_test  = minmax_normalize(x_test)   # Normalize the data to be between 0 and 1 using the min-max normalization

    x_train_features = feature_extraction(x_train, pca_comps_thresh)                  # Extract features from the images and align them in a dataframe

    x_test_features = feature_extraction(x_test, pca_comps_thresh)                                    # Extract features from the images and align them in a dataframe

    return x_train_features, y_train_encoded, x_test_features, y_test_encoded, le
    '''

    images, labels = sample_and_resize(dataset_path, n, size)  # Sample the images from the dataset path to the training path and testing path

    images = minmax_normalize(images)  # Normalize the data to be between 0 and 1 using the min-max normalization
    label, le = label_data_new(labels)  # Convert the labels to numbers for the model to be able to process them
    images_features = feature_extraction(images, pca_comps_thresh)  # Extract features from the images and align them in a dataframe

    x_train, x_test, y_train, y_test = train_test_split(images_features, label, test_size=1-ratio, random_state=42)  # Split the data into training and testing datasets

    return x_train, x_test, y_train, y_test, le


# This method receives an array of images paths, and returns a list of resized images a list of corresponding labels.
def sample_and_resize(data_path, n, size=128):

    images = []
    labels = []

    for artist_path in glob.glob(data_path+r'/*'):
        label = artist_path.split("\\")[-1]  # extracts the artist name from the directory path.

        artist_images = glob.glob(os.path.join(artist_path, "*.jpg"))   # list of images
        random.shuffle(artist_images)        # shuffle the images inside the images list

        # Adjust the number of images based on availability
        n = min(n, len(artist_images))

        # Partition the images to training and testing
        artist_images = artist_images[:n]

        for img_path in artist_images:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Read the image in color (BGR format)
            img = cv2.resize(img, (size, size))     # Resize the image to the given size, using weighted average values for interpolation.
            images.append(img)    # Add the processed image to the list of images
            labels.append(label)  # Add the label to the list of labels, matching the image in the same index in the images list.

    return np.array(images), np.array(labels)


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


def label_data_new(labels):
    """
    Converts the labels to numbers for the model to be able to process them, utilizing sklearn LabelEncoder.
    :param labels: The labels of the images in the dataset.
    :return: The encoded labels for the dataset.
    """
    # Converting the labels to numbers for the model to be able to process them.
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels_encoded = le.transform(labels)

    # Giving our data conventional names for easier use in the model.
    return labels_encoded, le


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


def vector_images(dataset):
    """
    Vectorizes the images in the dataset into 1D arrays.
    :param dataset: The dataset of images to be vectorized.
    :return: The vectorized images.
    """
    vectorized_images = []
    for img in dataset:
        # reduced_image = pca_reduction(img, pca_comps)
        vectorized_img = img.reshape(-1)
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
def gabor_images(filter_list, dataset):
    """
    Applies the Gabor filters to the images in the dataset.
    :param filter_list: The list of Gabor filters to be applied to the images.
    :param dataset: The dataset of images to be filtered.
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
            gabor_image = minmax_normalize(gabor_image)
            curr_gabor_images.append(gabor_image)

        gabor_images_dict[gabor_label] = np.array(curr_gabor_images)
        count += 1

    return gabor_images_dict


def sobel_images(kernel_sizes, dataset):
    """
    Applies the Sobel filter to the images in the dataset.
    :param kernel_sizes: The kernel size of the Sobel filter.
    :param dataset: The dataset of images to be filtered.
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
                channels.append(edge_sobel)                                                   # Add the filtered image to the list of filtered images

            edge_sobel_color = cv2.merge([channels[0], channels[1], channels[2]])      # Merge each color's magnitude with itself to create a 3-channel image (BGR format)
            edge_sobel_gray = cv2.merge([channels[3], channels[3], channels[3]])       # Merge the gray magnitude with itself to create a 3-channel image (BGR format)

            edge_sobel_gray = minmax_normalize(edge_sobel_gray)     # Normalize the gray  filtered image to be between 0 and 1 using the min-max normalization
            edge_sobel_color = minmax_normalize(edge_sobel_color)   # Normalize the color filtered image to be between 0 and 1 using the min-max normalization

            edge_sobel_color = edge_sobel_color.reshape(-1)  # Reshape the 3-channel image to be a 1D array
            edge_sobel_gray = edge_sobel_gray.reshape(-1)    # Reshape the 3-channel image to be a 1D array

            curr_sobel_color.append(edge_sobel_color)  # Add the color filtered image to the list of filtered images
            curr_sobel_gray.append(edge_sobel_gray)    # Add the gray  filtered image to the list of filtered images

        sobel_images_dict[sobel_label_color] = np.array(curr_sobel_color)  # Add the list of color filtered images to the dictionary of filtered images
        sobel_images_dict[sobel_label_gray] = np.array(curr_sobel_gray)    # Add the list of gray  filtered images to the dictionary of filtered images
        count += 1  # Increment the counter for the next label.

    return sobel_images_dict


def pca_reduction(image, pca_comps_threshold):
    """
    Reduces the image to the given number of components using PCA.
    :param image: The image to be reduced.
    :param pca_comps_threshold: The threshold for the number of components to reduce the image to.
    :return: The reduced image.
    """
    # If the image is colored, enters here
    if len(image.shape) == 3:
        b, g, r = cv2.split(image)  # Split the image into its BGR channels

        pca_b = PCA(n_components=pca_comps_threshold)  # Create a PCA object for the red channel
        pca_g = PCA(n_components=pca_comps_threshold)  # Create a PCA object for the green channel
        pca_r = PCA(n_components=pca_comps_threshold)  # Create a PCA object for the blue channel

        reduced_b = pca_b.fit_transform(b)  # Fit the PCA object to the red channel and transform it
        reduced_g = pca_g.fit_transform(g)  # Fit the PCA object to the green channel and transform it
        reduced_r = pca_r.fit_transform(r)  # Fit the PCA object to the blue channel and transform it

        reduced_image = cv2.merge([reduced_b, reduced_g, reduced_r])  # Merge the reduced channels to form the reduced image
        reduced_image = reduced_image  # Reshape the reduced image to be a 1D array

        return reduced_image

    # If the image is grayscale, enters here
    else:
        pca = PCA(n_components=pca_comps_threshold)
        reduced_image = pca.fit_transform(image)
        return reduced_image


# This function receives a flat image dataset, the original dataset of the images, and the pca threshold to reduce the dataset.
# It returns the reduced dataset.
def dataset_pca_reduction(dataset, original_dataset, threshold):
    """
    Reduces the dataset to the given number of components using PCA.
    :param dataset: The dataset to be reduced.
    :param original_dataset: The original shape of the images in the dataset.
    :param threshold: The number of components to reduce the dataset to.
    :return: The reduced dataset.
    """
    num_images, rows, cols, channels = original_dataset.shape  # Get the number of images, rows, columns, and channels in the dataset

    temp_dataset = dataset.reshape(num_images, rows, cols, channels)  # Reshape the dataset to its original shape
    dataset_b, dataset_g, dataset_r = [], [], []  # Lists to store the BGR channels of the dataset

    for image in temp_dataset:
        b_img, g_img, r_img = cv2.split(image)  # Split the image into its BGR channels
        dataset_b.append(b_img.reshape(-1))  # Add the blue channel to the list of blue channels
        dataset_g.append(g_img.reshape(-1))  # Add the green channel to the list of green channels
        dataset_r.append(r_img.reshape(-1))  # Add the red channel to the list of red channels

    max_components = 0  # A variable to store the maximum number of components that explain a variance above the threshold

    for channel in [dataset_b, dataset_g, dataset_r]:
        channel = np.array(channel)  # Convert the channel to a numpy array
        channel = channel.reshape(num_images, -1)  # Reshape the channel to be a 2D array, while rows: images and columns: pixels
        pca_channel = PCA(n_components=threshold)  # Create a PCA object for the channel
        pca_channel.fit_transform(channel)         # Fit the PCA object to the channel and transform it
        n_components = np.argmax(np.cumsum(pca_channel.explained_variance_ratio_) >= threshold) + 1  # Get the number of components that explain 99% of the variance

        if n_components > max_components:
            max_components = n_components

    reduced_dataset = []

    for channel in [dataset_b, dataset_g, dataset_r]:
        pca_channel = PCA(n_components=max_components)
        reduced_channel = pca_channel.fit_transform(channel)
        reduced_dataset.append(reduced_channel)

    reduced_dataset = cv2.merge(reduced_dataset)  # Merge the reduced channels to form the reduced dataset
    reduced_dataset = minmax_normalize(reduced_dataset)  # Normalize the reduced dataset to be between 0 and 1 using the min-max normalization
    reduced_dataset = reduced_dataset.reshape(num_images, -1)  # Reshape the reduced dataset to be a 1D array

    # Finally, reduce the dataset to the given threshold using PCA
    pca = PCA(n_components=threshold)
    reduced_dataset = pca.fit_transform(reduced_dataset)
    reduced_dataset = minmax_normalize(reduced_dataset)  # Normalize the reduced dataset to be between 0 and 1 using the min-max normalization

    return np.array(reduced_dataset)


# The following function is used to vectorize the images by extracting features from them, and aligning them in a dataframe.
# The input must be a 4 dimensional array. In our case, an array of colored images. Won't work with grayscale images.
def feature_extraction(dataset, pca_components_threshold):

    # The following parameters are used to create the Gabor and Sobel filters.
    f  = [0.1, 0.5]             # Represents the frequency of the sine component
    o  = [0, 90]                # Represents the orientation of the filter
    sa = [1.0]                  # Represents the spatial aspect ratio of the filter.
    sd = [0.5, 1.0]             # Represents the standard deviation of the filter
    p  = [0, np.pi/2]           # Represents the phase offset of the filter
    ks = [3, 5, 7]              # Represents the kernel size of the filter (K x K)

    filters = create_gabor_filters(f, o, sa, sd, p, ks)  # Create the Gabor filters based on the parameters above

    original_images = vector_images(dataset)               # Reduce the images to the given number of components using PCA
    gabor_images_dict = gabor_images(filters, dataset)     # Apply the Gabor filters to the images in the dataset
    sobel_images_dict = sobel_images(ks, dataset)          # Apply the Sobel filter to the images in the dataset

    # Reducing the images' y-axis components using PCA
    reduced_original_images = dataset_pca_reduction(original_images, dataset, pca_components_threshold)

    reduced_gabor_images_dict = {}
    for label, images_pixels in gabor_images_dict.items():
        reduced_gabor_images_dict[label] = dataset_pca_reduction(images_pixels, dataset, pca_components_threshold)

    reduced_sobel_images_dict = {}
    for label, images_pixels in sobel_images_dict.items():
        reduced_sobel_images_dict[label] = dataset_pca_reduction(images_pixels, dataset, pca_components_threshold)

    arrays_to_combine = [reduced_original_images]

    for label, images_pixels in reduced_gabor_images_dict.items():
        arrays_to_combine.append(images_pixels)

    for label, images_pixels in reduced_sobel_images_dict.items():
        arrays_to_combine.append(images_pixels)

    combined_array = np.hstack(arrays_to_combine)
    combined_df = pd.DataFrame(combined_array)

    return combined_df
