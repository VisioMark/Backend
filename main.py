import os
from fastapi import FastAPI
import pydantic
from image_module import ImageProcessing
from dir_module import image_dir_to_array
import tensorflow as tf
import numpy as np
import cv2

app = FastAPI()
model = tf.keras.models.load_model('saved_models/model_1.h5')

class ImageProcessingModel(pydantic.BaseModel):
    image_dir: str
    output_dir: str = ''
    no_of_questions: int = 40
    master_key: object = None

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post('/predict')
async def predict_score(ipm: ImageProcessingModel):
    image_processing = ImageProcessing(image_dir=ipm.image_dir, 
                                       output_dir=ipm.output_dir or ipm.image_dir,
                                       no_of_questions=ipm.no_of_questions, 
                                       master_key=ipm.master_key)
    image_processing.image_corruption_check()
    image_files = image_dir_to_array(image_dir=ipm.image_dir)
    
    prediction_list = []
    for image_file in image_files:
        image_processing.crop_rows(image_file=image_file)
    
        img = cv2.imread(os.path.join(ipm.output_dir, image_file))
        resize = tf.image.resize(img, [130, 20])
         # Make predictions
        predictions = model.predict(np.expand_dims(resize/255, axis=0))
        prediction_list.append(predictions.tolist())

        # Format the predictions and return them
    return {"predictions": prediction_list}
    
