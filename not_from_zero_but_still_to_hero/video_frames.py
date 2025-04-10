import cv2
 
cam = cv2.VideoCapture('./not_from_zero_but_still_to_hero/datasets/video_org.mp4')
 
currentframe = 0
  
while(True): 
      
    # reading from frame 
    ret,frame = cam.read()
    print(ret)
  
    if ret: 
        # if video is still left continue creating images 
        name = './not_from_zero_but_still_to_hero/datasets/frames/frame' + str(currentframe) + '.jpg'
        print ('Creating...' + name) 
  
        # writing the extracted images 
        cv2.imwrite(name, frame) 
  
        # increasing counter so that it will 
        # show how many frames are created 
        currentframe += 1
    else: 
        break
  
# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 