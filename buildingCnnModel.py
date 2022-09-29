from numpy import mean
from numpy import std
import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.layers import LeakyReLU
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report

#keras tensorflow içine taşındığı için bu şekilde import edilir
#convolutional layer-->pooling layer-->dense layer

#deneme ve eğitim datasetlerini yükleme
def loadDataset():
    #mnist dataseti yüklenir
    (trainX, trainY), (testX, testY) = mnist.load_data()
    plot(trainX)
    """
    trainX= eğitim verisi(resim)
    trainy=eğitim verisi için etiketler

    testX = test verisi(resim)
    tetsY = test verisi için etiketler
    """
    trainX = trainX.reshape((60000, 28, 28, 1)) #28x28 60000 tane resim ve resimler grayscale
    testX = testX.reshape((10000, 28, 28, 1))   #28x28 10000 tane resim ve resimler grayscale
    
    #to_categorical one hot encodingtir.
    #kategorik değişkenleri binary olarak temsil etmeye yarar.
    
    trainY = to_categorical(trainY,10)
    testY = to_categorical(testY,10)

    return trainX, trainY, testX, testY
""" 
def plot(trainX):
    plt.figure(figsize=(14,14))
    x,y= 10,4
    for i in range(40):
      plt.subplot(y,x,i+1)
      plt.imshow(trainX[i])
    plt.show()
"""
def pixels(train, test):
    # float'a çeviririz
    train2 = train.astype('float32')
    test2 = test.astype('float32')
    #eğitim ve test setlerinde rgb kullanılır. Bunu 0 ve 1 aralığına ayarlamak için 255 ile bölünür.
    train2 = train2 / 255.0
    test2= test2 / 255.0
    return train2, test2


# define cnn model
def define_model():
    model = Sequential()
    #model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(28,28,
    #filters=çıktı filtre sayısı,kernelsize=yüksekliği ve genişliği belirler
    #aktivasyon fonksiyonu
    #kernel-initilazer : kernel ağırlık matrisi için (initializer default glorot_uniform)

    model.add(Conv2D(64, (5, 5), activation='LeakyReLU', strides=(1,1) , kernel_initializer='he_uniform',input_shape=(28,28,1),padding="same"))
    #MaxPooling2D--> pool_size=(2, 2), strides=None, padding="valid", data_format=None
    model.add(MaxPooling2D((2, 2),2))
    
    #stride size kernel size'dan küçük olmalı
    #stride size---> kaydırma oranı
    model.add(Conv2D(32, (5, 5), activation='LeakyReLU',strides=(1,1) ,kernel_initializer='he_uniform',padding="same"))
    model.add(MaxPooling2D((2, 2),2))



    #16 nörondan oluşan ve relu fonksiyonu kullanan layer
    model.add(Conv2D(16, (3, 3), activation='LeakyReLU',strides=(1,1) ,kernel_initializer='he_uniform',padding="same"))
    model.add(MaxPooling2D((2, 2),2))
    model.add(Dropout(0.2))

    #bir layerda belirli bir oranın altında az kullanılan nöronu layerdan çıkarma -> drop out 

    #modeli düzleştirme
    #son katmana geçmeden önce 2D veriyi tek boyutlu hale getirmemiz gerek
    #bunun için -> Flatten()
    model.add(Flatten())

    #Dense---> units,activation=None,use_bias=True,kernel_initializer="glorot_uniform",bias_initializer="zeros",kernel_regularizer=None,
    #bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,
    #çıktı boyutsallığı,
    model.add(Dense(100, activation='LeakyReLU', kernel_initializer='he_uniform'))

    #softmax sınıflandırma yapar
    #10 etiket için 10 sinir ağı oluşturulur
    model.add(Dense(10, activation='softmax'))

    
    #SGD, öğrenme oranı ve momentumdan oluşur
    #learning_rate default=0.01
    #gradyan iyileştirmek için kullanılır
    opt = SGD(learning_rate=0.01, momentum=0.5)

    #optimizer,loss,metrics,loss_weights....
    #metrics; eğitim ve test sırasında modele göre evrilen metrik listesi categorical_crossentropy
    #loss fonksiyonu hataları minimize etmek için kullanılır.Hataları minimize  etmek için meansquared fonksiyonunu kullanır.
    #accuracy doğruluk oranlarının en efektif biçimde ortaya konulabilmesi için kullanılan başlıca metrik
    #optimizer tahmin ve loss fonksiyonunu karşılaştırarak, input weights optimize eder
    model.compile(optimizer=opt, loss='MeanSquaredError', metrics=['accuracy'])
    return model

def graphics(histories):
    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    print('\n')
    print('\n')
    print('\n')
    plt.show()

#epoch -> eğitim kaç iterasyonda gerçekleşecek
def evaluate_model(dataX, dataY):
    scores  = list()
    histories = list()
    # prepare cross validation
    #10 değeri dataseti 10'a böler ve her iterasyonda 1 dataset alt dalına inerek test eder.
    kfold = KFold(10, shuffle=True, random_state=1)
    #rastgele olarak (shuffle=True) veriyi test ve eğitim olarak ayırıyor
    for train_ix, test_ix in kfold.split(dataX):
        model = define_model()
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        #fit fonksiyonu;eğitim üzerinde,modeli değerlendirme işlemi yapar
        #küçük ve orta ölçekli datasetlerde modeli sığdırma işlemidir
        #x =MODEL EĞİTİM VERİSİ x, y = MODEL EĞİTİM VERİSİ
        #batch_size--->GRADYAN BAŞINA DİKKATE ALMASI GEREKEN ÖRNEK SAYISI,
        #batch size ---> model eğitilirken tüm veriler bir anda değil paket paket eğitilir
        #bu paketin içerisindeki veri sayısına batch size denir.
        #batch size = bir iterasyonda train örnekleri kullanım sayısı
        #batch size azaltılırsa, model çalışması hızlanır


        #epochs = MODELİ EĞİTMEK İÇİN GEREKLİ İTERASYON SAYISI,

        # validation_data = NULL,

        #verbose= 0 olursa ilerleme çubuğu gösterilmez, 1 olursa gösterilir
        #validation data = doğrulama verileridir. Modelin ideal olarak nerede eğitildiğini bulmaya yarar.
        history = model.fit(trainX, trainY, epochs=20, batch_size=32, validation_data=(testX, testY), verbose=0)
        #history değeri;eğitim sırasında kayıp değerlerini ve metrik değerlerini tutar

        #modeli değerlendirme
        #test verileri alınır
        #score=['loss', 'accuracy']
        #kayıp,doğruluk
        score= model.evaluate(testX, testY, verbose=1)

        """ zaten verbose 1 belirttiğim için bunu tekrar yazmama gerek yok
        print('LOSS %.2f' % (score[0]))
        print('ACCURACY %.2f' % (score[1]))
        """

        scores.append(score[1])
        histories.append(history)
    model.summary()
    return scores, histories


# summarize model performance
def summaryPerformance(scores):
    # ortalama,standart sapma,n sayısı
    print('\n\n\n')
    print('Accuracy: mean=%.2f std=%.2f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))



def runModel():
    # load dataset
    trainX, trainY, testX, testY = loadDataset()
    # prepare pixel data
    trainX, testX = pixels(trainX, testX)
    # değerlendirme
    scores, histories = evaluate_model(trainX, trainY)
    graphics(histories)
    # performans skorunu öğrenme
    #summarize_diagnostics(histories)
    summaryPerformance(scores)

# çalıştırma
if __name__ == "__main__":
    runModel()
    


