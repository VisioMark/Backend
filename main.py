import os
from fastapi import FastAPI, HTTPException, status
import pydantic
from __pred_methods__ import multiprocessing_predictions, serial_predictions
from dir_module import image_dir_to_array
import tensorflow as tf
import cv2
from fastapi.middleware.cors import CORSMiddleware
import logging


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=['GET', 'POST'],
    allow_headers=["*"],
)

# Configure the root logger
logging.basicConfig(level=logging.INFO)  # Set the desired log level

class ImageProcessingModel(pydantic.BaseModel):
    image_dir: str
    no_of_questions: str = '40'


@app.get("/")
def read_root():
    return {"Hello": "World"}


def image_corruption_check(image_dir):
    """This function will check if the image is corrupted or not.

    Args:
        image_dir (str): The path of the image directory
    """    
    
     # Validate the directory path
    if not os.path.isdir(image_dir):
        error_detail = f"Invalid directory path: `{image_dir}`"
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail= error_detail)
    
    image_files = image_dir_to_array(image_dir)
    for image_file in image_files:
        try:
            img = cv2.imread(os.path.join(image_dir, image_file))
            dummy = img.shape  # this line will throw the exception
        except:
            logging.error(" Image is not available or corrupted.")
            print(os.path.join(image_dir, image_file))
    logging.info("Image corruption check completed.")
    
    
@app.post("/predict")
async def predict_score(ipm: ImageProcessingModel):
    image_corruption_check(ipm.image_dir)
    
     # Validate the directory path
    if not os.path.isdir(ipm.image_dir):
        error_detail = f"Invalid directory path: `{ipm.image_dir}`"
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail= error_detail)
    
    image_file_names = image_dir_to_array(image_dir=ipm.image_dir)
    print(image_file_names)
    
    if len(image_file_names) <= 10:
        logging.info("Serial predictions started.")
        response = serial_predictions(ipm, image_file_names)
    
    if len(image_file_names) > 10:
        logging.info("Multiprocessing predictions started.")
        response = multiprocessing_predictions(ipm, image_file_names)
    
    if len(image_file_names) == 0:
        raise HTTPException(status_code=status.HTTP_200_SUCCESS, detail= "No images found in the directory.")
    
    return response





    # image_processing = ImageProcessing(
    #     image_dir=ipm.image_dir,
    #     no_of_questions=ipm.no_of_questions,
    #     master_key=ipm.master_key,
    # )
    # image_files = image_dir_to_array(image_dir=ipm.image_dir)
    # print(image_files)
    # prediction_list = {}
    # for image_file in image_files:
    #     image_processing.crop_rows(image_file=os.path.join(ipm.image_dir, image_file))

    #     # stripped_answers = image_dir_to_array(image_dir ='output_folder')
    #     # start_time = time.time()
    #     # for img in stripped_answers:
    #     #     imgs = cv2.imread(os.path.join('output_folder', img))
    #     #     resize = tf.image.resize(imgs, [130, 20])
    #     # Make predictions
    #     predictions = model.predict(np.expand_dims(resize/255, axis=0))
    #     char = show_equiv_label(predictions)
    #     prediction_list[img] = char
    #     # prediction_list.append(predictions.tolist())
    # end_time = time.time()

    # print("Time taken to predict: ", end_time - start_time)
    # Format the predictions and return them
    # return {"predictions": prediction_list}
