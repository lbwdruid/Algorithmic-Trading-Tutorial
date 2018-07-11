import pandas
import numpy
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt

CONST_TRAINING_SEQUENCE_LENGTH = 60
CONST_TESTING_CASES = 5

def dataNormalization(data):
	return [(datum-data[0])/data[0] for datum in data]

def dataDeNormalization(data, base):
	return [(datum+1)*base for datum in data]

def getDeepLearningData(ticker):
	# Step 1. Load data
	data = pandas.read_csv('../02. Data/01. IntradayUS/'+ticker+'.csv')['close'].tolist()

	# Step 2. Building Training data
	dataTraining = []
	for i in range(len(data)-CONST_TESTING_CASES*CONST_TRAINING_SEQUENCE_LENGTH):
		dataSegment = data[i:i+CONST_TRAINING_SEQUENCE_LENGTH+1]
		dataTraining.append(dataNormalization(dataSegment))

	dataTraining = numpy.array(dataTraining)
	numpy.random.shuffle(dataTraining)
	X_Training = dataTraining[:, :-1]
	Y_Training = dataTraining[:, -1]

	# Step 3. Building Testing data
	X_Testing = []
	Y_Testing_Base = []
	for i in range(CONST_TESTING_CASES, 0, -1):
		dataSegment = data[-(i+1)*CONST_TRAINING_SEQUENCE_LENGTH:-i*CONST_TRAINING_SEQUENCE_LENGTH]
		Y_Testing_Base.append(dataSegment[0])
		X_Testing.append(dataNormalization(dataSegment))

	Y_Testing = data[-CONST_TESTING_CASES*CONST_TRAINING_SEQUENCE_LENGTH:]

	X_Testing = numpy.array(X_Testing)
	Y_Testing = numpy.array(Y_Testing)

	# Step 4. Reshape for deep learning
	X_Training = numpy.reshape(X_Training, (X_Training.shape[0], X_Training.shape[1], 1))
	X_Testing = numpy.reshape(X_Testing, (X_Testing.shape[0], X_Testing.shape[1], 1))

	return X_Training, Y_Training, X_Testing, Y_Testing, Y_Testing_Base

def predict(model, X):
	predictionsNormalized = []

	for i in range(len(X)):
		data = X[i]
		result = []

		for j in range(CONST_TRAINING_SEQUENCE_LENGTH):
			predicted = model.predict(data[numpy.newaxis,:,:])[0,0]
			result.append(predicted)
			data = data[1:]
			data = numpy.insert(data, [CONST_TRAINING_SEQUENCE_LENGTH-1], predicted, axis=0)

		predictionsNormalized.append(result)

	return predictionsNormalized

def plotResults(Y_Hat, Y):
	plt.plot(Y)

	for i in range(len(Y_Hat)):
		padding = [None for _ in range(i*CONST_TRAINING_SEQUENCE_LENGTH)]
		plt.plot(padding+Y_Hat[i])

	plt.show()

def predictLSTM(ticker):
	# Step 1. Load data
	X_Training, Y_Training, X_Testing, Y_Testing, Y_Testing_Base = getDeepLearningData(ticker)

	# Step 2. Build model
	model = Sequential()

	model.add(LSTM(
		input_dim = 1,
		output_dim = 50,
		return_sequences=True))
	model.add(Dropout(0.2))

	model.add(LSTM(
		200,
		return_sequences=False))
	model.add(Dropout(0.2))

	model.add(Dense(output_dim=1))
	model.add(Activation('linear'))

	model.compile(loss='mse', optimizer='rmsprop')

	# Step 3. Train model
	model.fit(X_Training, Y_Training,
		batch_size=512,
		nb_epoch=5,
		validation_split=0.05)

	# Step 4. Predict
	predictionsNormalized = predict(model, X_Testing)

	# Step 5. De-nomalize
	predictions = []
	for i, row in enumerate(predictionsNormalized):
		predictions.append(dataDeNormalization(row, Y_Testing_Base[i]))

	# Step 6. Plot
	plotResults(predictions, Y_Testing)

predictLSTM(ticker='NVDA')