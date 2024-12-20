import os
import cv2
import glob
import numpy as np


def load_data_small():
    """
    This function loads images from the path: 'data/data_small' and returns the training
    and testing dataset. The dataset is a list of tuples where the first element is the 
    numpy array of shape (m, n) representing the image and the second element is its 
    classification (1 or 0).

    Returns:
        dataset: The first and second element represents the training and testing dataset respectively
    """
    # Begin your code (Part 1-1)
    # raise NotImplementedError("To be implemented")
    # Initialize lists to  store data
    dataset = []
    train_data = []
    test_data = []

    data_path = 'C:\\Users\\user\\Desktop\\AI_HW1\\HW1\\data\\data_small'
    train_path = data_path + '/train/'
    test_path = data_path + '/test/'
    root = str(train_path) + '/face/'
    for image in os.listdir(root):
        train_data.append((cv2.imread(root+image, cv2.IMREAD_GRAYSCALE), 1))
    
    root = str(train_path) + '/non-face/'
    for image in os.listdir(root):
        train_data.append((cv2.imread(root+image, cv2.IMREAD_GRAYSCALE), 0))
    
    root = str(test_path) + '/face/'
    for image in os.listdir(root):
        test_data.append((cv2.imread(root+image, cv2.IMREAD_GRAYSCALE), 1))
    
    root = str(test_path) + '/non-face/'
    for image in os.listdir(root):
        test_data.append((cv2.imread(root+image, cv2.IMREAD_GRAYSCALE), 0))
    
    dataset = (train_data, test_data)
    # End your code (Part 1-1)
    return dataset
    


def load_data_FDDB(data_idx="01"):
    """
        This function generates the training and testing dataset  form the path: 'data/data_FDDB'.
        The dataset is a list of tuples where the first element is the numpy array of shape (m, n)
        representing the image the second element is its classification (1 or 0).
        
        In the following, there are 4 main steps:
        1. Read the .txt file
        2. Crop the faces using the ground truth label in the .txt file
        3. Random crop the non-faces region
        4. Split the dataset into training dataset and testing dataset
        
        Parameters:
            data_idx: the data index string of the .txt file

        Returns:
            train_dataset: the training dataset
            test_dataset: the testing dataset
    """

    with open("C:\\Users\\user\\Desktop\\AI_HW1\\HW1\\data\\data_FDDB\\FDDB-folds\\FDDB-fold-{}-ellipseList.txt".format(data_idx)) as file:
        line_list = [line.rstrip() for line in file]

    # Set random seed for reproducing same image croping results
    np.random.seed(0)

    face_dataset, nonface_dataset = [], []
    line_idx = 0
    # Iterate through the .txt file
    # The detail .txt file structure can be seen in the README at https://vis-www.cs.umass.edu/fddb/
    while line_idx < len(line_list):
        img_gray = cv2.imread(os.path.join("C:\\Users\\user\\Desktop\\AI_HW1\\HW1\\data\\data_FDDB", line_list[line_idx] + ".jpg"), cv2.IMREAD_GRAYSCALE)
        num_faces = int(line_list[line_idx + 1])

        # Crop face region using the ground truth label
        face_box_list = []
        for i in range(num_faces):
            # Here, each face is denoted by:
            # <major_axis_radius minor_axis_radius angle center_x center_y 1>.
            coord = [int(float(j)) for j in line_list[line_idx + 2 + i].split()]
            x, y = coord[3] - coord[1], coord[4] - coord[0]            
            w, h = 2 * coord[1], 2 * coord[0]

            left_top = (max(x, 0), max(y, 0))
            right_bottom = (min(x + w, img_gray.shape[1]), min(y + h, img_gray.shape[0]))
            face_box_list.append([left_top, right_bottom])
            # cv2.rectangle(img_gray, left_top, right_bottom, (0, 255, 0), 2)

            img_crop = img_gray[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]].copy()
            face_dataset.append((cv2.resize(img_crop, (19, 19)), 1))

        line_idx += num_faces + 2

        # Random crop N non-face region
        # Here we set N equal to the number of faces to generate a balanced dataset
        # Note that we have already saved the bounding box of faces into face_box_list, 
        # you can utilize it for non-face region cropping
        for i in range(num_faces):
            # Randomly choose a non-face region
            img_height, img_width = img_gray.shape
            non_face_cropped = False
            while not non_face_cropped:
                # Randomly select a position for cropping
                start_x = np.random.randint(0, img_width - 19)
                start_y = np.random.randint(0, img_height - 19)
                end_x = start_x + 19
                end_y = start_y + 19
                
                # Check if the cropped region intersects with any face bounding box
                intersect = False
                for face_box in face_box_list:
                    face_left_top, face_right_bottom = face_box
                    face_start_x, face_start_y = face_left_top
                    face_end_x, face_end_y = face_right_bottom
                    if not (end_x < face_start_x or start_x > face_end_x or end_y < face_start_y
                            or start_y > face_end_y):
                        intersect = True
                        break
                
                # If no intersection, add non-face data
                if not intersect:
                    non_face_img = img_gray[start_y:end_y, start_x:end_x].copy()
                    nonface_dataset.append((cv2.resize(non_face_img, (19, 19)), 0))
                    non_face_cropped = True

        # cv2.imshow("windows", img_gray)
        # cv2.waitKey(0)

    # train test split
    num_face_data, num_nonface_data = len(face_dataset), len(nonface_dataset)
    SPLIT_RATIO = 0.7

    train_dataset = face_dataset[:int(SPLIT_RATIO * num_face_data)] + nonface_dataset[:int(SPLIT_RATIO * num_nonface_data)]
    test_dataset = face_dataset[int(SPLIT_RATIO * num_face_data):] + nonface_dataset[int(SPLIT_RATIO * num_nonface_data):]

    return train_dataset, test_dataset


def create_dataset(data_type):
    if data_type == "small":
        return load_data_small()
    else:
        return load_data_FDDB()
