import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report , accuracy_score , confusion_matrix


data = pd.read_csv("modeling.csv")
print(data)

x = data.iloc[:,3:13]
y = data.iloc[:,13]

geography = pd.get_dummies(x["Geography"] , drop_first=True)
gender = pd.get_dummies(x["Gender"] , drop_first=True)

print(geography)
print(gender)

x = pd.concat([x , geography , gender] , axis=1)

x = x.drop(['Geography' , 'Gender'] , axis=1)

print(x)
print(y)

train_x , test_x , train_y , test_y = train_test_split(x,y,test_size=0.2 , random_state=0)

sc = StandardScaler()
train_x = sc.fit_transform(train_x)


print(train_x)
print(test_x)

#making ANN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()

classifier.add(Dense(units = 6 ,  kernel_initializer= 'he_uniform' , activation='relu' , input_dim=11))
classifier.add(Dropout(0.3))
classifier.add(Dense(units=6 , kernel_initializer='he_uniform' , activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=1 , kernel_initializer='glorot_uniform' , activation='sigmoid'))



classifier.compile(optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'])


model_history = classifier.fit(train_x , train_y , validation_split=0.33 , batch_size=10 , epochs=50)


plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train' , 'test'])
plt.show()


plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train' , 'test'])
plt.show()


#prediction

pred_y = classifier.predict(test_x)
print(pred_y)
pred_y = (pred_y>0.5)
print(pred_y)


print(confusion_matrix(test_y , pred_y))
print(accuracy_score(test_y , pred_y))


