import os
from typing import Dict
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from utils import make_predictions

os.environ['CUDA_VISIBLE_DEVICES'] = ''


class ImageMarker:
    def __init__(self, image_path: str, no_of_questions: int, master_key: dict) -> None:
        self.image_path = image_path
        self.width = 1162
        self.height = 1600
        self.questions = no_of_questions
        
    def add_brightness(self, img: np.ndarray):
        """Add brightness and sharpness filter to the image

        Args:
            img (np.ndarray): _description_
        """
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        return img

    def load_diff_images(self, image_path):
        """This function helps you read, resize, and grayscale your images using TensorFlow

        Args:
            image_path (str): Path of the image to be read

        Returns:
            img: Original image as a TensorFlow tensor
            gray_img: Grayscale image as a TensorFlow tensor
            cnts: Contours
        """
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.width, self.height))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
        canny_img = cv2.Canny(img_blur, 75, 220)
        cnts = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return img, gray_img, cnts

    def process_image(self, image_path: str) -> np.ndarray:
        """this function helps you to preprocess your images and get gets
        the biggest contour

        Args:
            image (str): path of image from the crop_columns function

        Returns:
            paper: an array of the biggest contour
        """

        img, gray_img, cnts = self.load_diff_images(image_path=image_path)
        img_big_contour = gray_img.copy()

        self.add_brightness(img_big_contour)

        cnts = imutils.grab_contours(cnts)
        docCnt = None

        # ensure that at least one contour was found
        if len(cnts) > 0:
            # sort the contours according to their size in
            # descending order
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

            # loop over the sorted contours
            for c in cnts:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                # if our approximated contour has four points,
                # then we can assume we have found the shaded area
                if len(approx) == 4:
                    docCnt = approx
                    break

        # apply a four point perspective transform to both the
        # original image and grayscale image to obtain a top-down
        # view of the document
        paper = four_point_transform(img, docCnt.reshape(4, 2))
        warped = four_point_transform(gray_img, docCnt.reshape(4, 2))
        
        del img, gray_img, cnts, img_big_contour, docCnt, warped

        print("[INFO] Image preprocessing completed.")

        return paper

    def get_cropped_columns(self, image_path: str) -> list:
        """This function helps you to crop the columns of the image

        Args:
            image_file (str): Takes the path of the image from the crop_rows function

        Returns:
            list[(list, int)]: returns a list of tuples of the cropped columns with the start_question_number
        """
        paper = self.process_image(image_path=image_path)
        print("[INFO] Cropping columns...")
        # Resize the image just to get rid of the black border
        resize = paper[30 : paper.shape[0] - 25, 39 : paper.shape[1] - 33]

        selected_columns = []

        # Cropping the image into 5 columns
        if self.questions >= 1 or self.questions >= 40:
            first_col = resize[0 : resize.shape[0], 35:172]
            selected_columns.append((first_col, 1))
        if self.questions >= 41 or self.questions >= 80:
            second_col = resize[0 : resize.shape[0], 225:367]
            selected_columns.append((second_col, 41))
        if self.questions >= 81 or self.questions >= 120:
            third_col = resize[0 : resize.shape[0], 420:560]
            selected_columns.append((third_col, 81))
        if self.questions >= 121 or self.questions >= 160:
            fourth_col = resize[0 : resize.shape[0], 610:755]
            selected_columns.append((fourth_col, 121))
        if self.questions >= 161 or self.questions >= 200:
            fifth_col = resize[0 : resize.shape[0], 810:1100]
            selected_columns.append((fifth_col, 161))

        print("[INFO] Cropping columns completed.")
        

        return selected_columns

    def get_questions_data(self, image_path: str) -> None:
        """this function helps you to crop the rows of the image

        Args:
            image_file (array): path of the image
        """
        columns = self.get_cropped_columns(image_path=image_path)
        print("[INFO] Cropping rows...")

        questions_data = []
        for col_data, start_of_question in columns:
            # set the values
            count = 0
            r_delta = 0
            question = start_of_question
            for r in range(0, col_data.shape[0], 17):
                if question >= self.questions + 1:
                    break
                x = 20
                row = col_data[r + r_delta : r + r_delta + x, :]
                resize = tf.image.resize(row, [130, 20])
                questions_data.append(resize)

                count += 1
                question += 1
                if count % 5 == 0:  # Should jump r_delta times of pixels every 5th time
                    r_delta += 18
                if count == 40:
                    break

        questions_data = np.array(questions_data)
        return questions_data

    def predict_selections(self) -> Dict[int, str]:
        question_data = self.get_questions_data(self.image_path)
        predictions = make_predictions(question_data)
        results = {
            id + 1: predicted_label for id, predicted_label in enumerate(predictions)
        }
        return results
