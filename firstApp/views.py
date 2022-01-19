from django.shortcuts import render

from django.core.files.storage import FileSystemStorage

from keras.preprocessing import image
from tensorflow import keras


model = keras.models.load_model('./models/beansNet.v2')
img_height, img_width = 224, 224


def index(request):
    context = {'a':1}
    return render(request, 'index.html', context)


def predictImage(request):
    import numpy as np

    class_names = ['점무늬병', '불마름병', '정상']


    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)
    testimage = '.'+filePathName

    img = image.load_img(testimage, target_size=(img_height, img_width))

    prediction_scores = model.predict(np.expand_dims(img, axis=0))
    predicted_index = np.argmax(prediction_scores)

    context = {'filePathName': filePathName, 'predictedLabel': class_names[predicted_index]}
    return render(request, 'index.html', context)



def viewDataBase(request):
    import os
    listOfImages = os.listdir('./media/')
    listOfImagesPath = ['./media/'+i for i in listOfImages]
    context = {'listOfImagesPath': listOfImagesPath}
    return render(request, 'viewDB.html', context)