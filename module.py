import os
import cv2
import numpy as np
from imutils.perspective import four_point_transform


class ImageProcessing:
    """
    This class has functions to perform image processing.
    """
    def __init__(self, image_dir: str, output_dir: str, no_of_questions: int, master_key):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.width = 1162
        self.height = 1600
        self.questions = no_of_questions
        self.master_key = master_key
    
    def add_brightness(self, img: list):
        '''
        This function will add brightness to the image.
        '''
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
        return img
    
    def image_dir_to_array(self, image_dir: str):
        """ Return a list of all names of files in folder path
        
        Returns:
            list: list of all names of files in folder path
        """        
        return [f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f)) ]
        

    def image_corruption_check(self):
        '''
        This function will check if the image is corrupted or not.
        '''
        image_files = self.image_dir_to_array(image_dir=self.image_dir)
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
        img = cv2.imread(os.path.join(self.image_dir, image_file))
        img = cv2.resize(img, (self.width, self.height))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
        canny_img = cv2.Canny(img_blur, 75, 220)
        cnts = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return img, gray_img, cnts

    def image_preprocessing(self) -> list:        
        image_files = self.image_dir_to_array(image_dir=self.image_dir)
        for image_file in image_files:
            img, gray_img, cnts = self.load_diff_images(image_file)
            img_big_contour = gray_img.copy()

            self.add_brightness(img_big_contour)
            # cnts = cv2.findContours(Canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cnts = cv2.grab_contours(cnts)
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
            
            return paper
        
    def crop_columns(self)->dict:
        paper = self.image_preprocessing()
        # Resize the image just to get rid of the black border
        resize = paper[30:paper.shape[0] - 25, 39:paper.shape[1] - 33]
        
        selected_columns = {}
        # Cropping the image into 5 columns
        if self.questions <= 40:
            first_col = resize[0:resize.shape[0], 35: 172]
            selected_columns = {'first_col': first_col}
        if self.questions <= 80:
            second_col = resize[0:resize.shape[0], 225: 367]
            selected_columns['second_col'] = second_col
        if self.question <= 120:
            third_col = resize[0:resize.shape[0], 420: 560]
            selected_columns['third_col'] = third_col
        if self.question <= 160:
            fourth_col = resize[0: resize.shape[0], 610: 755]
            selected_columns['fourth_col'] = fourth_col
        if self.question <= 200:
            fifth_col  = resize[0:resize.shape[0], 810: 1100] 
            selected_columns['fifth_col'] = fifth_col
            
        return selected_columns
    
    def crop_rows(self) -> None:
        columns = self.crop_columns()
        
        output_folder = "output_folder"  # Set your desired output folder here
        
        # set the values
        count = 0
        r_delta = 0
        question = 1
        
        for col_name, col_data in columns.items():
            for r in range(0, col_data.shape[0], 17):
                x = 20
                for c in range(0, col_data.shape[1], 200):
                    row = col_data[r + r_delta: r + r_delta + x, c: c + 200]
                    file_folder = os.path.splitext(self.image_file)[0]
                    output_file_folder = os.path.join(output_folder, file_folder)
                    os.makedirs(output_file_folder, exist_ok=True)
                    cv2.imwrite(f"{output_file_folder}/Question{question}-{self.image_file}", row)
                    count += 1
                    question += 1
                    if count % 5 == 0:
                        r_delta += 18
                    if count == 40:
                        break
                if count == 40:
                    break
            if count == 40:
                break

        return 
            
            
            
            
if __name__ == '__main__': 
    image_processing = ImageProcessing(image_dir='', output_dir='output', no_of_questions=40)
    image_processing.image_corruption_check()
