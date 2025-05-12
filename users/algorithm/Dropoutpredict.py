
import pandas as pd
import numpy as np
class classification:
    data = pd.read_csv("media/Bloodlevel_dataset.csv")
    data = data.drop(["First_Name","Age","Gender",'avg'],axis=1)

    x = data.iloc[:,0:5]
    y = data.iloc[:,-1]
    #x = pd.get_dummies(x)

    from sklearn.model_selection import train_test_split
    #Splitting dataset into training  set and Test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    

    def DeepLearning(self):
        from keras.models import Sequential
        from keras.layers import Dense
        from matplotlib import pyplot as plt
        
        model = Sequential()
        model.add(Dense(4, input_dim = 5, activation = "relu"))
        model.add(Dense(3,activation = "relu" ))
        model.add(Dense(1, activation = "sigmoid"))
        model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
        
        
        print("x_train:", self.x_train)
        print("y_train", self.y_train)
        print("x_test", self.x_test)
        print("y_test", self.y_test)
        # X = np.asarray(x).astype(np.float32)
        history=model.fit(self.x_train, self.y_train, epochs = 5, batch_size = 32)
        #print("MY MODEL EVALUATE", model.evaluate(x, y))
        
        loss_and_metrics = model.evaluate(self.x_test, self.y_test)
        print("loss_and_metrics : ", loss_and_metrics)
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.legend()
        return loss_and_metrics

