

import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential
import matplotlib.pyplot as plt

##create gaussian
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def get_gaussian_train(x_values,mu,sig):
    summary_gausian = 0;
    for mu in[mu,mu+50,mu+100,mu+150]:
        summary_gausian =summary_gausian+ gaussian(x_values,mu,sig) # sig is 8
    return summary_gausian



#############################
###   Sin prediction ########
#############################
def build_nn_model2():
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(30, 3)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

mem_depth = 30
Train_index = 120
sample = 200

Fs = 120
f = 5
## create axis
x_axis = np.arange(sample)
#zeros = np.zeros(200)

x= x_axis
y1 = (np.sin(2 * np.pi * f * x / Fs) ) + 1.5*get_gaussian_train(x,20,8)

#Fs = 80
f = 9
y2 = (np.sin(2 * np.pi * f * x_axis / Fs +2) )

#Fs = 80
f = 15
x = np.arange(sample)
y3 = (np.sin(2 * np.pi * f * x_axis / Fs +3) )

y_target = (y1 + y2 + y3) / 3 - 1.5 * get_gaussian_train(x, 30, 8)

fig = plt.figure()
ax = fig.add_subplot(411)
plt.plot(x,y1)
ax = fig.add_subplot(412)
plt.plot(x,y2)
ax = fig.add_subplot(413)
plt.plot(x,y3)
ax = fig.add_subplot(414)
plt.plot(x, y_target)

plt.xlabel('sample(n)')
plt.ylabel('sine')

model = build_nn_model2()



X1_input = []
X2_input = []
X3_input = []
y_train = []
for i in range(mem_depth, Train_index):
    X1_input.append(y1[i - mem_depth:i])
    X2_input.append(y2[i - mem_depth:i])
    X3_input.append(y3[i - mem_depth:i])
    y_train.append(y_target[i-1])
X1_input, X2_input, X3_input, y_train = np.array(X1_input), np.array(X2_input), np.array(X3_input), np.array(y_train)

X_train = np.array([X1_input, X2_input, X3_input])#.reshape(Train_index - mem_depth, mem_depth, 3)

X_train2 = np.zeros([Train_index - mem_depth, mem_depth, 3])
for i in range(Train_index - mem_depth):
    for j in range(3):
        X_train2[i,:,j] = X_train[j,i,:]

history = model.fit(X_train2, y_train.reshape(Train_index - mem_depth, 1), epochs=100, validation_split=0.2, verbose=1)
#history = model.fit(X_train2,  epochs=100, validation_split=0.2, verbose=1)  Must have target data


#########################
#preparing test Data
#######################


X1_test = []
X2_test = []
X3_test = []
y_test = []
for i in range(Train_index  + mem_depth , sample):
    X1_test.append(y1[i - mem_depth:i])
    X2_test.append(y2[i - mem_depth:i])
    X3_test.append(y3[i - mem_depth:i])
    y_test.append(y_target[i-1])

X1_test, X2_test, X3_test, y_test = np.array(X1_test), np.array(X2_test), np.array(X3_test), np.array(y_test)

X_test = np.array([X1_test, X2_test, X3_test])#.reshape(X1_test.shape[0], mem_depth, 3)


X_test2 = np.zeros([X1_test.shape[0], mem_depth, 3])
for i in range(X1_test.shape[0]):
    for j in range(3):
        X_test2[i,:,j] = X_test[j,i,:]

test_output = model.predict(X_test2, verbose=0)
print(test_output)


fig1 = plt.figure()
ax = fig1.add_subplot(211)
plt.plot(x[Train_index + mem_depth:],y_test.reshape(sample - Train_index - mem_depth,1))
ax = fig1.add_subplot(212)
plt.plot(x[Train_index + mem_depth:],test_output)
