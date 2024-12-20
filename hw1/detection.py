import os
from unittest import result
import cv2
import utils
import numpy as np
import matplotlib.pyplot as plt


def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:A
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    # raise NotImplementedError("To be implemented")

    with open(dataPath, 'r') as file:
        lines = file.readlines()

    current_image = None
    for line in lines:
        parts = line.strip().split(' ')
        if len(parts) == 2:  # This is a new image
            if current_image is not None:
                # Display the result for the previous image before moving on to the next
                plt.figure()
                plt.imshow(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
                plt.show()
            # Load new image
            img_path = os.path.join('data/detect/', parts[0])  # Update with your correct folder path
            current_image = cv2.imread(img_path)
            num_faces = int(parts[1])
        elif len(parts) == 4 and current_image is not None:  # This is a face region
            x, y, w, h = map(int, parts)
            # Crop and resize face region to 19x19, then convert to grayscale
            face_region = cv2.resize(current_image[y:y+h, x:x+w], (19, 19))
            face_region_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Classify the face region
            is_face = clf.classify(face_region_gray)
            # Draw a green box if it's a face, red otherwise
            color = (0, 255, 0) if is_face else (0, 0, 255)
            cv2.rectangle(current_image, (x, y), (x+w, y+h), color, 2)

    # Display the result for the last image
    if current_image is not None:
        plt.figure()
        plt.imshow(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
        plt.show()

    # End your code (Part 4)
