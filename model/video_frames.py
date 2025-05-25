import cv2
 
video = cv2.VideoCapture('./not_from_zero_but_still_to_hero/datasets/video_org.mp4')
 
current_frame = 0
  
while(True): 
      
    # reading from frame 
    success, frame = video.read()
  
    if success: 
        # if video is still left continue creating images 
        name = './not_from_zero_but_still_to_hero/datasets/frames/frame' + str(current_frame) + '.jpg'
        print ('Creating...' + name) 
  
        # writing the extracted images 
        cv2.imwrite(name, frame) 
  
        # increasing counter so that it will 
        # show how many frames are created 
        current_frame += 1
    else: 
        break
  
# Release all space and windows once done 
video.release() 
cv2.destroyAllWindows() 