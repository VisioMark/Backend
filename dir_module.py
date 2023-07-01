import os
import tempfile

def image_dir_to_array(image_dir: str):
    """ Return a list of all names of images in folder path
    
    Returns:
        list: list of all names of images in folder path
    """
    for root, dirs, files in os.walk(image_dir):
        return files

def image_temp_dir():
    # Get the path of a temporary directory
    temp_dir = tempfile.gettempdir()

    # Generate a unique filename for the temporary image
    temp_filename = next(tempfile._get_candidate_names()) + ".jpg"
    temp_path = os.path.join(temp_dir, temp_filename)
