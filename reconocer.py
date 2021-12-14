import cv2 as cv
import os


dataPath = 'bank_data' 
#Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)


face_recognizer = cv.face.LBPHFaceRecognizer_create() 

# Leyendo el modelo
face_recognizer.read('modeloLBPHFace.xml')


cap = cv.VideoCapture(0)
#cap = cv.VideoCapture('Video.mp4')

faceClassif = cv.CascadeClassifier(cv.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
	ret,frame = cap.read()
	if ret == False: break
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	auxFrame = gray.copy()

	faces = faceClassif.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv.resize(rostro,(150,150),interpolation= cv.INTER_CUBIC)
		result = face_recognizer.predict(rostro)

		# LBPHFace
		if result[1] < 70:
			cv.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv.LINE_AA)
			cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		else:
			cv.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv.LINE_AA)
			cv.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
		
	cv.imshow('frame',frame)

	
	k = cv.waitKey(1)  
	if k == ord('q'): #cierra el programa al precionar q
		break

cap.release()
cv.destroyAllWindows()
