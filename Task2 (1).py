# Task 2 Machine lerning
import csv 
from random import seed
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plting

#-----------------------------------------------------
def convert_label(data):
    labels = [label[1] for label in data]
    labels_class = set(labels)
    lookup = {}
    for idx,value in enumerate(labels_class):
        lookup[value] = idx
    for i in range(len(labels)):
        labels[i] = lookup[labels[i]]
    return labels
#-----------------------------------------------------
def extract_features(dataset):
    features = []
    for row in dataset:
        features.append([float(feature) for feature in row[2:]])
    return features
#----------------------------------------------------------------
def load_dataset(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
#---------------------------------------------------------
def prepare_dataset(dataset):
    instances = extract_features(dataset)
    labels = convert_label(dataset)
    return(instances,labels)
#----------------------------------------------------
def compute_accuracy(actual_labels, predicted_labels):
    correct = 0
    for i in range(len(actual_labels)):
        if actual_labels[i] == predicted_labels[i]:
            correct += 1
    return correct / float(len(actual_labels)) * 100.0
#---------------------------------------------------------
def predict(instance, weights,bias):
    prediction = bias
    for i in range(len(instance)):
        prediction += weights[i] * instance[i]
        
    return 1.0 if prediction >= 0.0 else 0.0
#------------------------------------------------------
def predict_dataset(data,weights,bias):
    predicted =list()
    for instance in data:
        resualt =predict(instance, weights,bias)
        predicted.append(resualt)
    return predicted
#-------------------------------------------------------        
def train_weights(train_data,train_label, rate, n_epoch):
    weights = [0.0 for i in range(len(train_data[0]))]
    bias = 0.0
    for epoch in range(n_epoch):
        for idx,instance in enumerate(train_data):
            prediction = predict(instance, weights,bias)
            error = train_label[idx] - prediction
            bias = bias + rate * error
            for i in range(len(instance)):
                weights[i] = weights[i] + rate * error * instance[i]
    return (weights,bias)
#------------------------------------------------------------------
def train_one_epoch(train_data,train_label,l_rate,weights,bias):
        for idx,instance in enumerate(train_data):
            prediction = predict(instance, weights,bias)
            error = train_label[idx] - prediction
            bias = bias + l_rate * error
            for i in range(len(instance)):
                weights[i] = weights[i] + l_rate * error * instance[i]
        return (weights,bias)
#------------------------------------------------------------------------
def train_test_multiple_epoch(train_data,train_label,test_data,test_label, l_rate, n_epoch,weights,bias):
    train_accuracy = list()
    test_accuracy =list()
    y_predicted_train =list()
    y_predicted_test =list()
    for epoch in range(n_epoch):
        weights,bias = train_one_epoch(train_data,train_label,l_rate,weights,bias)
        y_train = predict_dataset(train_data,weights,bias)
        train_accuracy.append(compute_accuracy(train_label,y_train))
        y_test = predict_dataset(test_data,weights,bias)
        test_accuracy.append(compute_accuracy(test_label,y_test))
    return (weights,bias,train_accuracy,test_accuracy)
#----------------------------------------------------------------------
def perceptron(train_data, train_label, test_data,test_label, learining_rate, number_epoch):
    weights = [0.0 for i in range(len(train_data[0]))]
    bias = 0.0
    weights,bias,train_accuracy,test_accuracy = train_test_multiple_epoch(train_data,train_label,test_data,test_label, learining_rate, number_epoch,weights,bias)
    R_predicted = list()
    for exmple in test_data:
        R_predicted.append(predict(exmple, weights,bias))
    accuracy = compute_accuracy(R_predicted,test_label)
    return (accuracy,train_accuracy,test_accuracy)
#---------------------------------------------------------

#---------------------------------------------------
number_epoch = 4000
rate = 0.015
dataset = load_dataset("wdbc.data")
instances,lables= prepare_dataset(dataset)
trainX, testX, trainY, testY = train_test_split(
  instances, lables , random_state = 104, test_size = 0.2, shuffle = True)
acc,train, test= perceptron(trainX, trainY,testX, testY, rate, number_epoch)
epoch = [j for j in range(number_epoch)]
plting.plot(epoch,test,label="Test Accuracy")
plting.plot(epoch, train,label="Train Accuracy")
plting.xlabel("epoch")
plting.ylabel("accuracy")
plting.legend()
plting.show()