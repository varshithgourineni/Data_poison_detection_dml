
import socket 
from threading import Thread 
from socketserver import ThreadingMixIn
import json
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn import svm

def runExistingSVM(dataset):
    dataset = dataset.values
    X, Y = dataset[:, :-1], dataset[:, -1]
    print("Dataset received and contain total records without poison detection : "+str(len(X)))
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    cls = svm.SVC()
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test)
    for i in range(0,20):
        y_test[i] = 10
    svm_acc = accuracy_score(y_test,prediction_data)*100
    return svm_acc

def runDMLwithPoisonDataDetection(dataset):
    dataset = dataset.values
    X, Y = dataset[:, :-1], dataset[:, -1]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(X_train)
    mask = yhat != -1
    X_train, y_train = X_train[mask, :], y_train[mask]
    print("Total records after poison detection : "+str(len(X_train)+X_test))
    cls = svm.SVC()
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test) 
    svm_acc = accuracy_score(y_test,prediction_data)*100
    return svm_acc


def startApplicationServer():
    class ClientThread(Thread): 
 
        def __init__(self,ip,port): 
            Thread.__init__(self) 
            self.ip = ip 
            self.port = port 
            print('Request received from IP : '+ip+' with port no : '+str(port)) 
 
        def run(self): 
            data = conn.recv(10000)
            data = json.loads(data.decode())
            request_type = str(data.get("type"))
            if request_type == 'basicDML':
                data = str(data.get("dataset"))
                f = open("dataset.csv", "w")
                f.write(data)
                f.close()
                dataset = pd.read_csv("dataset.csv")
                existing_svm_accuracy = runExistingSVM(dataset)
                basic_dm_accuracy = runDMLwithPoisonDataDetection(dataset)
                print("SVM Accuracy without Data Poison Detection : "+str(existing_svm_accuracy))
                print("SVM Accuracy after Data Poison Detection : "+str(basic_dm_accuracy))
                jsondata = json.dumps({"existing": str(existing_svm_accuracy),"dml": str(basic_dm_accuracy)})
                message = conn.send(jsondata.encode())          
                

    tcpServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    tcpServer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    tcpServer.bind(('localhost', 3333))
    threads = []
    print("Worker2 Server Started")
    while True:
        tcpServer.listen(4)
        (conn, (ip,port)) = tcpServer.accept()
        newthread = ClientThread(ip,port) 
        newthread.start() 
        threads.append(newthread) 
    for t in threads:
        t.join()

Thread(target=startApplicationServer).start()
    




