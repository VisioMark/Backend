import os
from fastapi import FastAPI
import pydantic
from image_marker import ImageMarker
from dir_module import image_dir_to_array
import tensorflow as tf
import numpy as np
import cv2
import time

app = FastAPI()
model = tf.keras.models.load_model("saved_models/model_1.h5")


class ImageProcessingModel(pydantic.BaseModel):
    image_dir: str
    no_of_questions: int = 40
    master_key: dict = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


def image_corruption_check(image_dir):
    """
    This function will check if the image is corrupted or not.
    """
    image_files = image_dir_to_array(image_dir)
    for image_file in image_files:
        try:
            img = cv2.imread(os.path.join(image_dir, image_file))
            dummy = img.shape  # this line will throw the exception
        except:
            print("[INFO] Image is not available or corrupted.")
            print(os.path.join(image_dir, image_file))
    print("[INFO] Image corruption check completed.")


@app.post("/predict")
async def predict_score(ipm: ImageProcessingModel):
    # check for corrupted images
    image_corruption_check(ipm.image_dir)

    image_file_names = image_dir_to_array(image_dir=ipm.image_dir)
    print(image_file_names)

    response = {}

    for name in image_file_names:
        image_path = os.path.join(ipm.image_dir, name)
        image_marker = ImageMarker(
            image_path=image_path, no_of_questions=ipm.no_of_questions, master_key={}
        )

        start_timee = time.time()
        predictions = image_marker.predict_selections()
        end_time = time.time()

        print(f"time taken to predict {name}: {end_time - start_timee}")

        response[name] = predictions

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
