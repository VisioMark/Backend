import numpy as np
import cv2
from fastapi import HTTPException, status
import logging


def add_brightness(img: np.ndarray):
    """Add brightness and sharpness filter to the image

    Args:
        img (np.ndarray): _description_
    """
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)
    return img


def load_diff_images(image_path, width, height):
    """This function helps you read, resize, and grayscale your images using TensorFlow

    Args:
        image_path (str): Path of the image to be read

    Returns:
        img: Original image as a TensorFlow tensor
        gray_img: Grayscale image as a TensorFlow tensor
        cnts: Contours
    """
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (width, height))
        img = add_brightness(img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
        canny_img = cv2.Canny(img_blur, 75, 220)
    except:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Image not found"
        )

    return img, gray_img, canny_img


def find_contours(img: np.ndarray):
    """This function helps you to find the contours of the image

    Args:
        img (np.ndarray): Grayscale image as a TensorFlow tensor

    Returns:
        cnts: Contours
    """
    try:
        cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Could not find contours"
        )

    return cnts


def get_n_columns(questions: int, resized_img: np.ndarray):
    # Cropping the image into 5 columns

    selected_columns = []
    try:
        if questions >= 1 or questions >= 40:
            first_col = resized_img[0 : resized_img.shape[0], 35:172]
            selected_columns.append((first_col, 1))
    except:
        logging.error("Could not crop the first column")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not crop the first column",
        )
    try:
        if questions >= 41 or questions >= 80:
            second_col = resized_img[0 : resized_img.shape[0], 225:367]
            selected_columns.append((second_col, 41))
    except:
        logging.error("Could not crop the second column")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not crop the second column",
        )
    try:
        if questions >= 81 or questions >= 120:
            third_col = resized_img[0 : resized_img.shape[0], 420:560]
            selected_columns.append((third_col, 81))
    except:
        logging.error("Could not crop the third column")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not crop the third column",
        )
    try:
        if questions >= 121 or questions >= 160:
            fourth_col = resized_img[0 : resized_img.shape[0], 610:755]
            selected_columns.append((fourth_col, 121))
    except:
        logging.error("Could not crop the fourth column")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not crop the fourth column",
        )
    try:
        if questions >= 161 or questions >= 200:
            fifth_col = resized_img[0 : resized_img.shape[0], 810:1100]
            selected_columns.append((fifth_col, 161))
    except:
        logging.error("Could not crop the fifth column")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not crop the fifth column",
        )

    logging.info("Cropping columns completed.")

    return selected_columns


def get_all_cropped_index_number(resized_image: np.ndarray):
    combined_images = []
    firstCol = resized_image[3 : resized_image.shape[0] - 3, 3:55]
    combined_images.append(firstCol)
    secondCol = resized_image[3 : resized_image.shape[0] - 3, 63:115]
    combined_images.append(secondCol)

    thirdCol = resized_image[3 : resized_image.shape[0] - 3, 125:180]
    combined_images.append(thirdCol)

    fourthCol = resized_image[3 : resized_image.shape[0] - 3, 185:240]
    print("fourth", fourthCol)
    combined_images.append(fourthCol)

    fifthCol = resized_image[3 : resized_image.shape[0] - 3, 248:300]
    combined_images.append(fifthCol)

    sixthCol = resized_image[3 : resized_image.shape[0] - 3, 310:360]
    combined_images.append(sixthCol)

    seventhCol = resized_image[3 : resized_image.shape[0] - 3, 370:-5]
    combined_images.append(seventhCol)

    return combined_images
