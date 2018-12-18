import numpy as np
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

# Generate data
X = np.random.randint(0, 100, 100000, dtype=np.int32)
y = np.zeros(shape=(X.shape[0], 4), dtype=np.int32)

# Init scaler
scaler = MinMaxScaler()

# Wait, didn't we solve FizzBuzz here..
# This really requires better dataset balancing.
# All categories should have equal representation, 
# but here we heavily lean towards [any num] => [0, 0, 0, 0]
cnt = 0
for i, val in enumerate(X):
    if val % 3 == 0 and val % 5 == 0:
        y[i] = np.array([0, 0, 0, 1])
    elif val % 5 == 0 and cnt % 5 == 0:
        y[i] = np.array([0, 0, 1, 0])
    elif val % 3 == 0 and cnt % 3 == 0:
        y[i] = np.array([0, 1, 0, 0])
    elif cnt % 15 == 0:
        y[i] = np.array([1, 0, 0, 0])
    cnt += 1

# Train test split, very important
idx = int(0.8 * len(X))
X_train = X[:idx]
X_test = X[idx:]
y_train = y[:idx]
y_test = y[idx:]

# Scaler requires 2d
X_train = X_train.reshape(-1, 1)
X_train = scaler.fit_transform(X_train)
X_train = X_train.reshape(-1,)

X_test = X_test.reshape(-1, 1)
X_test = scaler.transform(X_test)
X_test = X_test.reshape(-1,)

# 3 hidden layers is technically Deep Learning..
input = Input(shape=(1,))
x = Dense(12, activation='relu')(input)
x = Dense(12, activation='relu')(input)
x = Dense(12, activation='relu')(input)
output = Dense(4, activation='softmax')(x)

# use the alpha optimizer nadam
model = Model(inputs=input, outputs=output)
model.compile(optimizer='nadam',
              loss='categorical_crossentropy')
model.fit(X_train, y_train, batch_size=128, epochs=20)

X_test = X_test.reshape(-1, 1)
X_test = scaler.inverse_transform(X_test)
X_test = X_test.reshape(-1,)

# Get preds and convert to category
preds = model.predict(X_test)
preds = np.argmax(preds, axis=1)
y_test = np.argmax(y_test, axis=1)

# Convert category to string and print result
for pred, actual, inval in zip(preds, y_test, X_test):
    # print actual/inval if you want
    if pred == 3:
        print('fizzbuzz')
    elif pred == 2:
        print('buzz')
    elif pred == 1:
        print('fizz')
    else:
        pass
