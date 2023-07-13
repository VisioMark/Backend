import os
import logging
import tensorflow as tf
import cv2
import numpy as np
import imutils
from typing import Dict
from fastapi import HTTPException, status
from imutils.perspective import four_point_transform
from helpers.utils import make_predictions, mark_predictions
from helpers.__img_utils__ import load_diff_images_for_idx_no, load_diff_images_for_shading , get_n_columns, find_contours, get_all_cropped_index_number

os.environ['CUDA_VISIBLE_DEVICES'] = ''


class ImageMarker:
    def __init__(self, image_path: str, no_of_questions: str, master_key: dict) -> None:
        self.image_path = image_path
        self.width = 1162
        self.height = 1600
        self.questions = int(no_of_questions)
        self.master_key = master_key
        
    def start_shading_processing(self) -> tuple([np.ndarray, np.ndarray, np.ndarray]):
        """This function starts the processing of the image

        Returns:
            img, gray_img, canny_img: Original image, Grayscale image, Canny image
        """
        diff_images_for_shading = load_diff_images_for_shading(image_path=self.image_path, width=self.width, height=self.height)
        
        
        return diff_images_for_shading
    
    def start_indx_processing(self) -> tuple([np.ndarray, np.ndarray, np.ndarray]):
        """This function starts the processing of the image

        Returns:
            img, gray_img, canny_img: Original image, Grayscale image, Canny image
        """
        diff_images_for_idx = load_diff_images_for_idx_no(image_path=self.image_path, width=self.width, height=self.height)
        
        
        return diff_images_for_idx
    
    def process_image_for_shading(self,diff_images) -> np.ndarray:
        """this function helps you to preprocess your images and get gets
        the biggest contour

        Args:
            image (str): path of image from the crop_columns function

        Returns:
            paper: an array of the biggest contour
        """

        img , gray_img, canny_img = diff_images
        img_big_contour = gray_img.copy()

        cnts = find_contours(canny_img)
        try:
            cnts = imutils.grab_contours(cnts)
        except:
            logging.error("Could not grab contours.")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail= "Error occuring during image processing: grabbing contours")
        
        doc_cnt = None

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
                    doc_cnt = approx
                    break
        else:
            logging.error("No contours found.")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail= "Error occuring during image processing: finding contours")

        # apply a four point perspective transform to both the
        # original image and grayscale image to obtain a top-down
        # view of the document
        paper = four_point_transform(img, doc_cnt.reshape(4, 2))
        warped = four_point_transform(gray_img, doc_cnt.reshape(4, 2))
        
        del gray_img, cnts, img_big_contour, doc_cnt, warped

        logging.info("Image preprocessing completed.")

        return paper
                  
    def get_index_no(self, diff_images) -> np.ndarray:
        """This function gets a canny(ied) image and processes it to get the section
        for the index number.

        Args:
            img (np.ndarray): Canny image
        """        
        img, gray_img, canny_img, resized_img = diff_images
               
        # # Resize the original image to get just the area around the index number.
        # resized_img = img[10:img.shape[0]//3, 30:img.shape[1]//3]

        contours = find_contours(canny_img)[0]

        guided_image = resized_img.copy()

        biggest_cnt = None
        for cnt in range(0, len(contours)):
            if cv2.contourArea(contours[cnt]) > 1000:
                print(cv2.contourArea(contours[cnt]))
                biggest_cnt = cnt
                cv2.drawContours(guided_image, contours, cnt, (255, 0, 0), 10)

        cnt = contours[biggest_cnt]
        
        # Find the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(cnt)

        # Crop the image based on the contour
        idxno_image = resized_img[y:y+h, x:x+w]


        # REMOVE [n] PIXELS FORM EACH SIDE
        resized_image = idxno_image[80:155, 125:idxno_image.shape[1] - 4]
        
        combined_images = get_all_cropped_index_number(resized_image=resized_image)
        imgs = np.array([self.preprocess_idx_img(img) for img in combined_images])
        # img = np.invert(resized_image)
        # plt.imshow(img, cmap='gray')
        # Reshape and normalize the image
        # img = np.expand_dims(img, axis=-1)  # Add an extra axis for the single channel
        # img = np.expand_dims(img, axis=0)  # Add batch dimension
        # img = np.apply_along_axis(lambda img: img/255.0, 1, imgs)
        # # img = img / 255.0  # Normalize the pixel values

        # # Resize the image using tf.image.resize
        # img = tf.image.resize(img, [28, 28])
        return imgs
        
    def preprocess_idx_img(self, img:np.ndarray) -> np.ndarray:
        """This function helps you to preprocess the index number image

        Args:
            img (np.ndarray): The index number image

        Returns:
            np.ndarray: The preprocessed index number image
        """
        if img.size == 0:
            raise ValueError("Empty image provided.")
        img = np.invert(img)
        img = img.astype(np.float32) / 255.0  # Normalize the pixel values

        if img.shape[0] == 0 or img.shape[1] == 0:
            raise ValueError("Invalid image dimensions.")

        # Resize the image using tf.image.resize
        img = tf.image.resize(img, [28, 28])
        return img
    

    def get_cropped_columns(self, diff_images) -> list:
        """This function helps you to crop the columns of the image

        Returns:
            list[(list, int)]: returns a list of tuples of the cropped columns with the start_question_number
        """
        paper = self.process_image_for_shading(diff_images=diff_images)
        print("[INFO] Cropping columns...")
        # Resize the image just to get rid of the black border
        try:
            resize = paper[30 : paper.shape[0] - 25, 39 : paper.shape[1] - 33]
        except Exception as exc:
            logging.error(f"Could not resize the big contour image.- error: {exc}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail= "Error occuring during image processing: resizing image")
        
        
        selected_columns = get_n_columns(questions=self.questions, resized_img=resize)
        # selected_columns = []

        # # Cropping the image into 5 columns
        # try:
        #     if self.questions >= 1 or self.questions >= 40:
        #         first_col = resize[0 : resize.shape[0], 35:172]
        #         selected_columns.append((first_col, 1))
        # except:
        #     logging.error('Could not crop the first column')
        #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not crop the first column")
        # try:
        #     if self.questions >= 41 or self.questions >= 80:
        #         second_col = resize[0 : resize.shape[0], 225:367]
        #         selected_columns.append((second_col, 41))
        # except:
        #     logging.error('Could not crop the second column')
        #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not crop the second column")
        # try:
        #     if self.questions >= 81 or self.questions >= 120:
        #         third_col = resize[0 : resize.shape[0], 420:560]
        #         selected_columns.append((third_col, 81))
        # except:
        #     logging.error('Could not crop the third column')
        #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not crop the third column")
        # try:
        #     if self.questions >= 121 or self.questions >= 160:
        #         fourth_col = resize[0 : resize.shape[0], 610:755]
        #         selected_columns.append((fourth_col, 121))
        # except:
        #     logging.error('Could not crop the fourth column')
        #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not crop the fourth column")
        # try:
        #     if self.questions >= 161 or self.questions >= 200:
        #         fifth_col = resize[0 : resize.shape[0], 810:1100]
        #         selected_columns.append((fifth_col, 161))
        # except:
        #     logging.error('Could not crop the fifth column')
        #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not crop the fifth column")

        # logging.info("Cropping columns completed.")
        

        return selected_columns

    def get_questions_data(self, diff_images ) -> np.ndarray:
        """this function helps you to crop the rows of the image
        """
        columns = self.get_cropped_columns(diff_images=diff_images)
        logging.info("Cropping rows...")

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
        """This function helps you to make predictions

        Returns:
            Dict[int, str]: returns a dictionary of the file name, predictions and score
        """
        diff_images = self.start_shading_processing()
        question_data = self.get_questions_data(diff_images=diff_images)
        
        idx_diff_images = self.start_indx_processing()
        index_number = self.get_index_no(diff_images=idx_diff_images)
        
        predictions = make_predictions(shading_arr=question_data, idx_num_arr=index_number )
        results = {
            id + 1: predicted_label for id, predicted_label in enumerate(predictions)
        }
        score = mark_predictions(results, self.master_key)
        accum_result = {
            "file_name": (self.image_path).split('/')[-1],
            'predictions': results,
            "score": score
        }
        return accum_result
