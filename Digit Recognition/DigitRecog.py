#!/usr/bin/env python
# coding: utf-8

# # Data Prep,Training and evaluation

# In[6]:


import cv2
import numpy as np


image = cv2.imread('./digits.png')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
small = cv2.pyrDown(image)
#cv2.imshow('Digits image',small)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



# Splitting Image
# Gives 4D array of 50 x 100 x 20 x 20
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]


x = np.array(cells)
print ("The shape of our cells array : "+ str(x.shape))


# splitting entire data into 2 sets : Training (70% ) Data set (30%)
train = x[:,:70].reshape(-1,400).astype(np.float32)
test = x[:,70:100].reshape(-1,400).astype(np.float32)


# Creating Labels
k = [0,1,2,3,4,5,6,7,8,9]
train_labels = np.repeat(k,350)[:,np.newaxis]
test_labels = np.repeat(k,150)[:,np.newaxis]


knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE, train_labels)
ret, result, neighbors, distance = knn.findNearest(test,k=3)


# now checking accuracy

matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * (100.0 / result.size )
print ("Accuracy is = %.2f" % accuracy + "%" ) 


# # Defining functions to input img

# In[7]:


import cv2
import numpy as np

def x_cord_contour (contour):
    
    if cv2.contourArea(contour) > 10:
        M = cv2.moments(contour)
        return (int(M['m10']/M['m00']))
    return 0 # Check this line
    
def makeSquare (not_square):
    # This func takes an image and makes dimension square
    BLACK = [0,0,0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]
    
    # print("Height = ",height," Width = ",width)
    
    if(height == width):
        square = not_square
        return square
    
    else:
        doublesize = cv2.resize(not_square,(2*width,2*height),interpolation = cv2.INTER_CUBIC)
        height = height * 2
        width = width *2
        
        if(height > width):
            pad = (int((height-width)/2))
            doublesize_square = cv2.copyMakeBorder(doublesize,0,0,pad,pad,cv2.BORDER_CONSTANT,value = BLACK)
        
        else:
            pad = (int((width - height)/2))
            doublesize_square = cv2.copyMakeBorder(doublesize,pad,pad,0,0,cv2.BORDER_CONSTANT,value = BLACK)
          
        doublesize_square_dim = doublesize_square.shape
        
        return doublesize_square
    
    
    
def resize_to_pixel (dimensions,image ):
    # resizing to specified dimensions
        
    buffer_pix = 4
    dimensions = dimensions - buffer_pix
    squared = image
    r = float(dimensions)/squared.shape[1]
    dim = (dimensions, int(squared.shape[0]*r)) 
    resized = cv2.resize (image, dim, interpolation = cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
        
    BLACK = [0,0,0]
        
    if (height_r > width_r ):
            resized = cv2.copyMakeBorder (resized,0,0,0,1,cv2.BORDER_CONSTANT,value = BLACK)
        
    if (height_r < width_r):
            resized = cv2.copyMakeBorder (resized,1,0,0,0,cv2.BORDER_CONSTANT,value = BLACK)
        
        
    p = 2
        
    ReSizedImg = resized = cv2.copyMakeBorder (resized,p,p,p,p,cv2.BORDER_CONSTANT,value = BLACK)
    img_dim = ReSizedImg.shape
        
    height = img_dim[0]
    width = img_dim[1]
        
    return ReSizedImg
            
            
            


# # Loading an Image and Preprocessing

# In[29]:


import cv2
import numpy as np

def recogniseDigits ( image ):
    #image = cv2.imread('./test8.jfif')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray",gray)
    #cv2.waitKey(0)

    # Blurring Image
    blurred = cv2.GaussianBlur(gray,(5,5),0)
    #cv2.imshow("blurred",blurred)
    #cv2.waitKey(0)

    edged = cv2.Canny(blurred, 30, 150)
    #cv2.imshow("edged",edged)
    #cv2.waitKey(0)

    # Fint Contours 
    contours, _ = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sorting contours left to right w.r.t x coord
    contours = sorted(contours, key = x_cord_contour, reverse = False)

    # Empty array to store entire no.
    full_number = []

    # Loop over Contours
    for c in contours:
        # Computing bounding rectangle
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.drawContours(image,contours, -1, (0,255,0), 3)
        #cv2.imshow("Contours", image)

        # Filtering the contour
        if w>=5 and h>=25:  
            roi = blurred[y:y+h, x:x+w]
            ret, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
            squared = makeSquare(roi)
            final = resize_to_pixel (20, squared)
            #cv2.imshow("final",final)
            final_array = final.reshape((1,400))
            final_array = final_array.astype(np.float32)
            ret, result, neighbors, distance = knn.findNearest(final_array, k=1)
            number = str(int(float(result[0])))
            full_number.append(number)

            # draw rectangle around digit + show digit

            cv2.rectangle(image, (x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(image, number, (x, y+20 ), 
                        cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0), 2)
            #cv2.imshow("image",image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
    
    cv2.imshow("image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print ("The Number is: "+ ''.join(full_number))


    
        
        
    
    


# In[32]:


# import the necessary packages
import cv2
import os, os.path
 

 
#image path and valid extensions
imageDir = "./testImages" #specify your path here
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jfif"] #specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]
 
#create a list all files in directory and
#append files with a vaild extention to image_path_list
for file in os.listdir(imageDir):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(imageDir, file))
 
#loop through image_path_list to open each image
for imagePath in image_path_list:
    image = cv2.imread(imagePath)
    
    # display the image on screen with imshow()
    # after checking that it loaded
    if image is not None:
        #cv2.imshow(imagePath, image)
        recogniseDigits(image)
    elif image is None:
        print ("Error loading: " + imagePath)
        #end this loop iteration and move on to next image
        continue
    
    # wait time in milliseconds
    # this is required to show the image
    # 0 = wait indefinitely
    # exit when escape key is pressed
    key = cv2.waitKey(0)
    if key == 27: # escape
        break
 
# close any open windows
cv2.destroyAllWindows()


# In[ ]:




