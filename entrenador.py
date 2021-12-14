import cv2 as cv 
import os
import numpy as np

dataPath = 'bank_data'
#Cambia a la ruta donde hayas almacenado la  Data
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
	personPath = dataPath + '/' + nameDir
	print('Leyendo las imágenes')

	for fileName in os.listdir(personPath):
		print('Rostros: ', nameDir + '/' + fileName)
		labels.append(label)
		facesData.append(cv.imread(personPath+'/'+fileName,0))
	label = label + 1

# Método para entrenar el reconocedor
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Entrenando el reconocedor de rostros
print("Entrenando ->->->")
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido
face_recognizer.write('modeloLBPHFace.xml')
print("Modelo almacenado  ! Finish")