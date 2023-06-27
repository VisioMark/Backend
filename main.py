import os
from fastapi import FastAPI
import pydantic
from image_module import ImageProcessing
from dir_module import image_dir_to_array
from utils import show_equiv_label
import tensorflow as tf
import numpy as np
import cv2

app = FastAPI()
model = tf.keras.models.load_model('saved_models/model_1.h5')

class ImageProcessingModel(pydantic.BaseModel):
    image_dir: str
    no_of_questions: int = 40
    master_key: dict = None

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post('/predict')
async def predict_score(ipm: ImageProcessingModel):
    image_processing = ImageProcessing(image_dir=ipm.image_dir, 
                                       no_of_questions=ipm.no_of_questions, 
                                       master_key=ipm.master_key)
    image_processing.image_corruption_check()
    image_files = image_dir_to_array(image_dir=ipm.image_dir)
    print(image_files)
    prediction_list = {}
    for image_file in image_files:
        image_processing.crop_rows(image_file= os.path.join(ipm.image_dir, image_file))
    
        stripped_answers = image_dir_to_array(image_dir ='output_folder')
        print(stripped_answers)
        for img in stripped_answers:
            imgs = cv2.imread(os.path.join('output_folder', img))
            resize = tf.image.resize(imgs, [130, 20])
            # Make predictions
            predictions = model.predict(np.expand_dims(resize/255, axis=0))
            char = show_equiv_label(predictions)
            prediction_list[img] = char
            # prediction_list.append(predictions.tolist())

        # Format the predictions and return them
    return {"predictions": prediction_list}
    
