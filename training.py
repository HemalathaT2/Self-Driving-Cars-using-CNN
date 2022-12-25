print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import *
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg


path = 'myData'
data = importDataInfo(path) 

data = balanceData(data,display=False)


imagesPath, steering = loadData(path,data)
print(imagesPath[0],steering[0])

xTrain,xVal,yTrain,yVal = train_test_split(imagesPath,steering,test_size=0.2,random_state=5)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

model = creatModel()
model.summary()

history = model.fit(batchGen(xTrain,yTrain,100,1),steps_per_epoch=300,epochs=10,validation_data=batchGen(xVal,yVal,100,0),
                     validation_steps=200)

model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
