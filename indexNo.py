from dir_module import image_dir_to_array
import os
import cv2

image_dir = image_dir_to_array('exam_sheets')

for image in image_dir:
    img = cv2.imread(os.path.join('exam_sheets', image))
    resized_img = img[300:img.shape[0]//3, 30:img.shape[1]//3]
    imgGray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
    imgCanny = cv2.Canny(imgBlur,10,70) # APPLY CANNY 
    
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)# FIND ALL CONTOURS

    guided_image = resized_img.copy()

    biggestCnt = None
    for cnt in range(0, len(contours)):
        if cv2.contourArea(contours[cnt]) > 1000:
            biggestCnt = cnt
            cv2.drawContours(guided_image, contours, cnt, (255, 0, 0), 10)

    cnt = contours[biggestCnt]
    
    # Find the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(cnt)

    # Crop the image based on the contour
    idxno_image = resized_img[y:y+h, x:x+w]


    # REMOVE [n] PIXELS FORM EACH SIDE
    resized_image = idxno_image[70:idxno_image.shape[0] - 5, 125:idxno_image.shape[1] - 4] 
    
    # Crop out to get all individual columns

    os.makedirs('stripped_indx', exist_ok=True)
    firstCol = resized_image[90:resized_image.shape[0], 0:60]
    cv2.imwrite(f"stripped_indx/{image}-firstcol.jpg", firstCol)
    
    secondCol = resized_image[90:resized_image.shape[0], 60:120]
    cv2.imwrite(f"stripped_indx/{image}-secondcol.jpg", secondCol)

    thirdCol = resized_image[90:resized_image.shape[0], 120:180]
    cv2.imwrite(f"stripped_indx/{image}-thid.jpg", thirdCol)

    fourthCol = resized_image[90:resized_image.shape[0], 180:240]
    cv2.imwrite(f"stripped_indx/{image}-fourth.jpg", fourthCol)

    fifthCol = resized_image[90:resized_image.shape[0], 240:300]
    cv2.imwrite(f"stripped_indx/{image}-fifth.jpg", fifthCol)

    sixthCol = resized_image[90:resized_image.shape[0], 300:360]
    cv2.imwrite(f"stripped_indx/{image}-sixth.jpg", sixthCol)

    seventhCol = resized_image[90:resized_image.shape[0], 360:]
    cv2.imwrite(f"stripped_indx/{image}-seventh.jpg", seventhCol)
