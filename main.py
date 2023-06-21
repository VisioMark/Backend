import os
from fastapi import FastAPI
import pydantic
from module import ImageProcessing
import tensorflow as tf
import numpy as np
import cv2

app = FastAPI()
model = tf.keras.models.load_model('model/model.h5')

class ImageProcessingModel(pydantic.BaseModel):
    image_dir: str
    output_dir: str = image_dir
    no_of_questions: int = 40
    master_key: object = None

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get('/predict')
async def predict_score(IPM: ImageProcessingModel):
    image_processing = ImageProcessing(image_dir=IPM.folder, 
                                       output_dir=IPM.folder, 
                                       no_of_questions=IPM.num_of_questions, 
                                       master_key=IPM.master_key)
    image_processing.image_corruption_check()
    image_processing.crop_rows()
    
    image_files = image_processing.image_dir_to_array(image_dir=IPM.output_dir)
    
    for image_file in image_files:
        img = cv2.imread(os.path.join(IPM.output_dir, image_file))
        resize = tf.image.resize(img, [130, 20])
         # Make predictions
        predictions = model.predict(np.expand_dims(resize/255, axis=0))

        # Format the predictions and return them
        return {"predictions": predictions.tolist()}
    
    return