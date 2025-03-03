
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import re
import numpy as np 
import pandas as pd
import socket
import json
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split

main = tkinter.Tk()
main.title("Data Poison Detection Schemes for Distributed Machine Learning")
main.geometry("1300x1200")

global filename
global svm_acc,basic_acc,semi_acc
global part1,part2
global first,second

def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def divideDataset():
    global first,second
    global part1,part2
    global header
    text.delete('1.0', END)
    part1 = ''
    part2 = ''
    header = ''
    dataset = pd.read_csv(filename)
    divide_size = len(dataset)
    text.insert(END,"Dataset contains total records : "+str(divide_size)+"\n")
    p1 = divide_size / 2
    p2 = p1
    if (p1+p2) != divide_size:
        p2 = p2 + 1
    count = 0
    text.insert(END,"Worker1 divided dataset total records : "+str(p1)+"\n")
    text.insert(END,"Worker2 divided dataset total records : "+str(p2)+"\n")
    with open(filename, "r") as file:
        for line in file:
            if len(header) == 0:
                header = str(line)
                print(header)
            else:
                line = line.strip('\n')
                line = line.strip()
                if count < p1:
                    part1+=line+"\n"
                else:
                    part2+=line+"\n"
                count = count + 1    
    file.close()
    first = header+part1
    second = header+part2
    


def runBasic():
    global first,second
    text.delete('1.0', END)
    global svm_acc,basic_acc
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    client.connect(('localhost', 2222))
    jsondata = json.dumps({"type":"basicDML","dataset": first})
    message = client.send(jsondata.encode())
    data = client.recv(1000)
    data = json.loads(data.decode())
    svm_acc = float(str(data.get("existing")))
    basic_acc = float(str(data.get("dml")))
    text.insert(END,"Existing SVM Accuracy Received from Worker1 : "+str(svm_acc)+"\n")
    text.insert(END,"DML SVM Accuracy Received from Worker1 : "+str(basic_acc)+"\n")

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    client.connect(('localhost', 3333))
    jsondata = json.dumps({"type":"basicDML","dataset": first})
    message = client.send(jsondata.encode())
    data = client.recv(1000)
    data = json.loads(data.decode())
    svm_acc1 = float(str(data.get("existing")))
    basic_acc1 = float(str(data.get("dml")))
    text.insert(END,"Existing SVM Accuracy Received from Worker2 : "+str(svm_acc1)+"\n")
    text.insert(END,"DML SVM Accuracy Received from Worker2 : "+str(basic_acc1)+"\n")

    svm_acc = svm_acc + svm_acc1
    basic_acc = basic_acc + basic_acc1
    svm_acc = svm_acc / 2
    basic_acc = basic_acc / 2
    text.insert(END,"Existing SVM Total Accuracy : "+str(svm_acc)+"\n")
    text.insert(END,"DML SVM Total Accuracy      : "+str(basic_acc)+"\n")

def runSemi():
    global semi_acc
    dataset = pd.read_csv(filename)
    dataset = dataset.values
    X, Y = dataset[:, :-1], dataset[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(X_train)
    mask = yhat != -1
    X_train, y_train = X_train[mask, :], y_train[mask]
    cls = svm.SVC()
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test) 
    semi_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"Semi-DML Accuracy      : "+str(semi_acc)+"\n")


    
def graph():
    height = [svm_acc,basic_acc,semi_acc]
    bars = ('Existing SVM Accuracy', 'Basic DML Accuracy','Semi DML Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def close():
    main.destroy()
    
font = ('times', 14, 'bold')
title = Label(main, text='Data Poison Detection Schemes for Distributed Machine Learning')
title.config(bg='yellow3', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

divideButton = Button(main, text="Divide Dataset", command=divideDataset)
divideButton.place(x=50,y=150)
divideButton.config(font=font1) 

basicButton = Button(main, text="Distribute Dataset & Run Basic-DML", command=runBasic)
basicButton.place(x=310,y=150)
basicButton.config(font=font1) 

semi = Button(main, text="Run Semi-DML", command=runSemi)
semi.place(x=650,y=150)
semi.config(font=font1) 

graphbutton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphbutton.place(x=50,y=200)
graphbutton.config(font=font1) 

exitb = Button(main, text="Exit", command=close)
exitb.place(x=310,y=200)
exitb.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='burlywood2')
main.mainloop()
