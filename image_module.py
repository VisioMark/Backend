import os
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from dir_module import image_dir_to_array


class ImageProcessing:
    """
    This class has functions to perform image processing.
    """
    def __init__(self, image_dir: str, no_of_questions: int, master_key: dict):
        self.image_dir = image_dir
        self.width = 1162
        self.height = 1600
        self.questions = no_of_questions
        self.master_key =  master_key
        
        print(no_of_questions)
    
    def add_brightness(self, img: list):
        '''
        This function will add brightness and sharpness filter to the image.
        '''
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
        return img
    
   

    def image_corruption_check(self):
        '''
        This function will check if the image is corrupted or not.
        '''
        image_files = image_dir_to_array(self.image_dir)
        for image_file in image_files:
            try:
                img = cv2.imread(os.path.join(self.image_dir, image_file))
                dummy = img.shape # this line will throw the exception
            except:
                print("[INFO] Image is not available or corrupted.")
                print(os.path.join(self.image_dir, image_file))
        print("[INFO] Image corruption check completed.")
        
    def load_diff_images(self, image_file) :
        """this function helps you read, resize and grayscale your images

        Args:
            image_file (image_file): image to be read

        Returns:
            img: Original image
            gray_img: Grayscale image
            cnts: Contours
        """        
        img = cv2.imread(image_file)
        img = cv2.resize(img, (self.width, self.height))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
        canny_img = cv2.Canny(img_blur, 75, 220)
        cnts = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return img, gray_img, cnts

    def image_preprocessing(self, image:str) -> list:
        """this function helps you to preprocess your images and get gets 
        the biggest contour

        Args:
            image (str): path of image from the crop_columns function

        Returns:
            paper: an array of the biggest contour
        """
        img, gray_img, cnts = self.load_diff_images(image_file=image)
        img_big_contour = gray_img.copy()

        self.add_brightness(img_big_contour)
        # cnts = cv2.findContours(Canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        warped = four_point_transform(gray_img, docCnt.reshape(4,2))
        # os.makedirs(self.output_dir, exist_ok=True)
        # cv2.imwrite(os.path.join(self.output_dir, image_file), paper)
        
        print("[INFO] Image preprocessing completed.")
        return paper
        
    def crop_columns(self, image_file:str)-> list:
        """This function helps you to crop the columns of the image

        Args:
            image_file (str): Takes the path of the image from the crop_rows function

        Returns:
            list[(list, int)]: returns a list of tuples of the cropped columns with the start_question_number
        """        
        paper = self.image_preprocessing(image=image_file)
        print("[INFO] Cropping columns...")
        # Resize the image just to get rid of the black border
        resize = paper[30:paper.shape[0] - 25, 39:paper.shape[1] - 33]
        
        selected_columns = []
        
        # Cropping the image into 5 columns
        if self.questions>=1 or self.questions >= 40:
            first_col = resize[0:resize.shape[0], 35: 172]
            selected_columns.append((first_col, 1))
        if  self.questions >=41 or self.questions >= 80:
            second_col = resize[0:resize.shape[0], 225: 367]
            selected_columns.append((second_col, 41))
        if self.questions >=81 or self.questions >= 120:
            third_col = resize[0:resize.shape[0], 420: 560]
            selected_columns.append((third_col, 81))
        if self.questions >= 121 or self.questions >= 160:
            fourth_col = resize[0: resize.shape[0], 610: 755]
            selected_columns.append((fourth_col, 121))
        if self.questions >= 161 or self.questions >= 200:
            fifth_col  = resize[0:resize.shape[0], 810: 1100] 
            selected_columns.append((fifth_col, 161))
        
        print("[INFO] Cropping columns completed.")  
        return selected_columns
    
    def crop_rows(self, image_file: str) -> None:
        """this function helps you to crop the rows of the image

        Args:
            image_file (array): path of the image
        """        
        columns = self.crop_columns(image_file=image_file)
        print("[INFO] Cropping rows...")
        output_folder = "output_folder"  # Set your desired output folder here
        
        
        for col_data, start_of_question in columns:
            # set the values
            count = 0
            r_delta = 0
            question = start_of_question
            for r in range(0, col_data.shape[0], 17):
                if question >= self.questions + 1:
                    break
                x = 20
                for c in range(0, col_data.shape[1], 200):
                    row = col_data[r + r_delta: r + r_delta + x, c: c + 200]
                    os.makedirs(output_folder, exist_ok=True)
                    cv2.imwrite(f"{output_folder}/Question{question}.jpg", row)
                    count += 1
                    question += 1
                    if count % 5 == 0: # Should jump r_delta times of pixels every 5th time
                        r_delta += 18
                    if count == 40:
                        break
                if count == 40:
                    break            
           
if __name__ == '__main__': 
    image_processing = ImageProcessing(image_dir='test', no_of_questions=60, master_key={})
    image_processing.crop_rows(image_file='./test/001.jpg')