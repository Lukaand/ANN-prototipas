# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import math
import csv
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys


# %%

# failai kuriuos norime transformuoti
names = [r'Raw_Data\02-14-2018.csv']

# Po kiek eluciu isimti is failo
Ch_size = 500000

# ar pirmas nuskaitymas
head = True

# for ciklo pradzia
for n in names:
    # nuskaitome po chunka
    for chunk in pd.read_csv(n, chunksize=Ch_size, low_memory = False):
        # isiemame visus NAN ir INF
        chunk = chunk[~chunk.isin([np.nan, np.inf, -np.inf]).any(1)]
        # isiemame laiko stulpeli
        chunk = chunk.drop(['Timestamp'], axis=1)


        # isrenkame stulpeliu pavadinimus
        if head == True:
            columns = chunk.columns.values

        # atskiriam kur yra reiksmes kur yra tipas
        X = chunk.iloc[:, 0:-1].values
        y = chunk.iloc[:, -1].values

        # suzimime kur yra normalus ivykis (0) kur ataka (1)
        suby = []
        for i in range(len(y)):
            if y[i] == 'Benign' or y[i] == 'Normal':
                suby.append(0)
            else:
                suby.append(1)
        y = suby.copy()
        del suby
    
        # suskirstomai train ir test datasetus su kuriais bus mokinamas ir testuojamas ANN
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

        # datasetai yra tranformuojami. Nezinau ar cia as gerai darau, bet kolkas geresnio budo nezinau
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # sudedami atgal i dataframus
        training_dataset = np.column_stack((X_train, y_train))
        test_dataset = np.column_stack((X_test, y_test))

        training_dataset = pd.DataFrame(data=training_dataset)
        test_dataset = pd.DataFrame(data=test_dataset)

        # sudedami stulpeliu pavadinimai
        training_dataset.columns = columns
        test_dataset.columns = columns


        # training_dataset, test_dataset = train_test_split(chunk, test_size = 0.2, random_state = 10)

        # idedti i .csv failus
        if head:
            test_dataset.to_csv(r'Temp_Data\test_data.csv')
            training_dataset.to_csv(r'Temp_Data\train_data.csv')
            head = False
        else:
            test_dataset.to_csv(r'Temp_Data\test_data.csv', mode='a', header=False)
            training_dataset.to_csv(r'Temp_Data\train_data.csv', mode='a', header=False)


