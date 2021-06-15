from re import T
import numpy as np
import pandas as pd
import tensorflow as tf
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def failo_paruosimas(name):
    X = []
    y = []

    for N in name:
        dataset = pd.read_csv(N)
        X.extend(dataset.iloc[:, 7:-1].values)
        y.extend(dataset.iloc[:, -1].values)
        print(X[0])


    # ohe = OneHotEncoder(categories='auto')
    # ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,3,6])], remainder='passthrough')
    # X = np.array(ct.fit_transform(X))
    # print(X[0])

    subX = []
    suby = []
    for i in range(len(y)):
        if y[i] == "Normal":
            subX.append(X[i].copy())
            suby.append(0)
        if y[i] == "Attack":
            subX.append(X[i].copy())
            suby.append(1)
    X = subX.copy()
    y = suby.copy()
    del subX
    del suby
    subX = []
    suby = []
    Yup = False
    for i in range(len(X)):
        for j in range(len(X[0])):
            if math.isnan(X[i][j]) or math.isinf(X[i][j]):
                Yup = True
                break
        if Yup == False:
            subX.append(X[i].copy())
            suby.append(y[i])
        Yup = False     
    X = subX.copy()
    y = suby.copy()
    del subX
    del suby

    good  = 0
    bad = 0
    for i in y:
        if i == 0:
            good = good + 1
        elif i == 1:
            bad = bad + 1
    print(good,bad)
    return X,y,good,bad

def Ratio(X,y,good,bad,r):
    if bad < good:
        f = round(good/(r*bad))
        g = 0
    else:
        f = round(bad/(r*good))
        g = 1
    hh = 0
    subX = []
    suby = []
    for i in range(len(y)):
        if y[i] == g:
            hh = hh + 1
            if hh == f:
                subX.append(X[i].copy())
                suby.append(y[i])
                hh = 0
        else:
            subX.append(X[i].copy())
            suby.append(y[i])
    X = subX.copy()
    y = suby.copy()
    del subX
    del suby
    good  = 0
    bad = 0
    for i in y:
        if i == 0:
            good = good + 1
        elif i == 1:
            bad = bad + 1
    print(good,bad)
    return X,y


def ANN(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
    return ann,X_test,y_test

def TEST(ann,X_test,y_test):
    y_pred = ann.predict(X_test)
    y_pred = (y_pred > 0.5)
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(accuracy_score(y_test, y_pred))
    accuracy_score(y_test, y_pred)
    
    print("normaliu variantu buvo",cm[0][0]+cm[0][1],". Correct responses", cm[0][0],"Increct", cm[0][1],"prob for normal", cm[0][0]/(cm[0][0]+cm[0][1]))
    print("Ataku variantu buvo",cm[1][0]+cm[1][1],". Correct responses", cm[1][1],"Increct", cm[1][0],"prob for atack", cm[1][1]/(cm[1][0]+cm[1][1]))
    

# def just_test(X,y,ann):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1, random_state = 0)
#     print(X[0],len(X[0]))
#     print(X_test[0],len(X_test[0]))
    
#     # X_train = np.asarray(X_train)
#     # y_train = np.asarray(y_train)
#     X_test = np.asarray(X)
#     y_test = np.asarray(y)

#     sc = StandardScaler()
#     # X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)

#     y_pred = ann.predict(X_test)
#     y_pred = (y_pred > 0.5)
#     np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
#     cm = confusion_matrix(y_test, y_pred)
#     print(cm)
#     print(accuracy_score(y_test, y_pred))

def attack_only(X,y):
    subX = []
    suby = []
    good  = 0
    bad = 0
    print("-----")
    for i in range(len(y)):
        if y[i] == 1:
            subX.append(X[i].copy())
            suby.append(y[i])
    X = subX.copy()
    y = suby.copy()
    del subX
    del suby

    good  = 0
    bad = 0
    for i in y:
        if i == 0:
            good = good + 1
        elif i == 1:
            bad = bad + 1
    print(good,bad)

    return X,y


names = [r'C:\Users\lukut\Desktop\Darbas\data\testbed-12jun.pcap_Flow.csv',r'C:\Users\lukut\Desktop\Darbas\data\testbed-13jun.pcap_Flow.csv',r'C:\Users\lukut\Desktop\Darbas\data\testbed-14jun.pcap_Flow.csv',r'C:\Users\lukut\Desktop\Darbas\data\testbed-15jun.pcap_Flow.csv',r'C:\Users\lukut\Desktop\Darbas\data\testbed-16jun.pcap_Flow.csv',r'C:\Users\lukut\Desktop\Darbas\data\testbed-17jun.pcap_Flow.csv']


X,y,good,bad = failo_paruosimas(names)
X,y = Ratio(X,y,good,bad,3)



# # X = subX.copy()
# # y = suby.copy()

# # del subX
# # del suby

ann,X_test,y_test = ANN(X,y)
TEST(ann,X_test,y_test)

# X,y,good,bad = failo_paruosimas(r'C:\Users\lukut\Desktop\Darbas\data\testbed-13jun.pcap_Flow.csv')
# X,y = Ratio(X,y,good,bad,3)
# ann,X_test,y_test = ANN(X,y)
# print("testo score (13)")
# TEST(ann,X_test,y_test)


# X,y,good,bad = failo_paruosimas(r'C:\Users\lukut\Desktop\Darbas\data\testbed-15jun.pcap_Flow.csv')
# print("testo score (15)")
# # X,y = attack_only(X,y)
# ann = 0
# just_test(X,y,ann)
# X,y = attack_only(X,y)
# just_test(X,y,ann)

# X,y,good,bad = failo_paruosimas(r'C:\Users\lukut\Desktop\Darbas\data\testbed-14jun.pcap_Flow.csv')
# print("testo score (14)")
# # X,y = attack_only(X,y)
# just_test(X,y,ann)
# X,y = attack_only(X,y)
# just_test(X,y,ann)

# X,y,good,bad = failo_paruosimas(r'C:\Users\lukut\Desktop\Darbas\data\testbed-15jun.pcap_Flow.csv')
# print("testo score (15)")
# # X,y = attack_only(X,y)
# just_test(X,y,ann)
# X,y = attack_only(X,y)
# just_test(X,y,ann)

# X,y,good,bad = failo_paruosimas(r'C:\Users\lukut\Desktop\Darbas\data\testbed-16jun.pcap_Flow.csv')
# print("testo score (16)")
# # X,y = attack_only(X,y)
# just_test(X,y,ann)
# X,y = attack_only(X,y)
# just_test(X,y,ann)

# X,y,good,bad = failo_paruosimas(r'C:\Users\lukut\Desktop\Darbas\data\testbed-17jun.pcap_Flow.csv')
# print("testo score (17)")
# # X,y = attack_only(X,y)
# just_test(X,y,ann)
# X,y = attack_only(X,y)
# just_test(X,y,ann)


