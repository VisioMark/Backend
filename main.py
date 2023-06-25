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
    output_dir: str = ''
    no_of_questions: int = 40
    master_key: object = None

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post('/predict')
async def predict_score(ipm: ImageProcessingModel):
    image_processing = ImageProcessing(image_dir=ipm.folder, 
                                       output_dir=ipm.output_dir or ipm.image_dir,
                                       no_of_questions=ipm.num_of_questions, 
                                       master_key=ipm.master_key)
    image_processing.image_corruption_check()
    image_processing.crop_rows()
    
    image_files = image_processing.image_dir_to_array(image_dir=ipm.output_dir)
    
    prediction_list = []
    for image_file in image_files:
        img = cv2.imread(os.path.join(ipm.output_dir, image_file))
        resize = tf.image.resize(img, [130, 20])
         # Make predictions
        predictions = model.predict(np.expand_dims(resize/255, axis=0))
        prediction_list.append(predictions.tolist())

        # Format the predictions and return them
    return {"predictions": prediction_list}
    
