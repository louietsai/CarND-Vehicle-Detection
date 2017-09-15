import cv2
#import cv2.cv

cap = cv2.VideoCapture('./project_video.mp4')
frame_width=1280
frame_height=720
fourcc = cv2.cv.CV_FOURCC(*'MPEG')
out = cv2.VideoWriter('project_video_output.avi',fourcc, 30, (frame_width,frame_height))

print("before while")
count = 0
while cap.isOpened():
    print("before read")
    ret,frame = cap.read()
    print("after read ret:",ret)
    #if cv2.waitKey(10) & 0xFF == ord('q'):
    #    break
    try:
        cv2.imshow('window-name',frame)
    except:
        break
    cv2.waitKey(1)
    filepath = "demo_images/"+"file_"+str(count)+".png"
    print("filepath : ", filepath)
    #cv2.imwrite(filepath,frame)
    count += 1
for i in range(1252):
    filepath = "proc_images/"+"file_"+str(i)+".png"
    print("filepath : ", filepath)
    image = cv2.imread(filepath)
    print("filepath : ", filepath)
    try:
        cv2.imshow('window-name',image)
    except:
        break
    cv2.waitKey(1)
    out.write(image)

cap.release()
out.release()
#cap.destroyAllWindows()
