import cv2
import time
videoName="output41.avi"
#video içe aktarma
cap=cv2.VideoCapture(videoName)
print("width: ",cap.get(3))
#get fonksiyonunun 3.indeksi genişliği verir

print("heigth: ",cap.get(4))
#get fonksiyonunun 4.indeksi yüksekliği verir
if cap.isOpened()== False:
    print("error")

#frame, videonun içindeki her bir resim

#learning fps
fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS: {0}".format(fps))

i=0
a=0
while True:
    ret, frame=cap.read()
    if ret==True:
         #time.sleep(0.4)
        cv2.imshow("video", frame)
        if(a%5==0):
            cv2.imwrite('frame'+str(i)+'.png',frame)
            i+=1
        a += 1
    else:
        break
    if cv2.waitKey(1) &0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

