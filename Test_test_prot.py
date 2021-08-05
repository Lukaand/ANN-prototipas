# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
from itertools import permutations,combinations,product
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier


# %%
# cia yra generatorius. Tai yra funkcija kuri paimą tam tikrą eilučių kiekį "row_count" is csv. failo kad neuzkrautum atminties.
# Taip pat, visa informacija informacia yra sudedam yra tam tikru "batchu" kiekius. Tai yra daroma del to, kad mokymas butu geresnis
def generator(File, batch_size):
    # paimamu eiluciu skaicius
    row_count = 50000
    # suskaiciuojama kiek bus naudojama eiluciu per epocha. Epocha yra vienas ciklas per kuri yra idedami visos eilutes.
    samples_per_epoch = sum(len(row) for row in pd.read_csv(File,chunksize=row_count))
    # batchu kiekis per epocha
    number_of_batches = samples_per_epoch/batch_size
    # suskaiciuojama po kiek reikia suskirstyti chunkus, kad is ju gautusi sveikas batchu skaicius  
    chunk_size_divided = int(row_count/batch_size)
    # pradedamas dalinimas
    counter = 0
    while 1:
        # isemamas chunkas is .csv failo. 
        # pvz: eluciu kiekis yra 120 chunku dydis yra 50. Tai is chunko bus isemami 3 chunkai su dydziais 50,50,20
        for chunk in pd.read_csv(File, chunksize=chunk_size_divided*batch_size):
            # jei chunko dydis yra toks pats koks yra nustatyta. 
            if len(chunk) == chunk_size_divided*batch_size:
                # suskirsto chunku stulpelius 
                X_data = chunk.iloc[:, 0:-1].values
                y_data = chunk.iloc[:, -1].values

                # chunkai suskirstomi i batchus
                for i in range(chunk_size_divided):

                    X_batch = np.array(X_data[batch_size*i:batch_size*(i+1)]).astype('float32')
                    y_batch = np.array(y_data[batch_size*i:batch_size*(i+1)]).astype('float32')
                    counter += 1
                    # batchai idedami i ann
                    yield X_batch,y_batch
            # jeigu failo pabaiga ir chunkas yra mazesnis negu numatyta
            else:
                # jei chunkas nesuskirsto tolygiai i batchus. paimamas likutis
                if len(chunk)/batch_size == int(len(chunk)/batch_size):
                    chunk_size_divided_ending = int(len(chunk)/batch_size)
                else:
                    chunk_size_divided_ending = int(len(chunk)/batch_size) + 1

                for i in range(chunk_size_divided_ending):
                    X_batch = np.array(X_data[batch_size*i:batch_size*(i+1)]).astype('float32')
                    y_batch = np.array(y_data[batch_size*i:batch_size*(i+1)]).astype('float32')
                    counter += 1
                    yield X_batch,y_batch
            #restart counter to yeild data in the next epoch as well

        if counter >= number_of_batches:
            counter = 0


# %%
# sudaromas ann modelis. siuo metu 3 layeriai po 5 nodes
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=5, activation='relu'))
ann.add(tf.keras.layers.Dense(units=5, activation='relu'))
ann.add(tf.keras.layers.Dense(units=5, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# suvedamas batcho dydis
batch_size = 8

# suskaiciuojama kiek eiluciu turi datasetas
# .csv failai jau yra transformuoti. dabar as naudojau tik 02-14-2018.cvs dataseta.
# failas buvo tranformuotas su Transform.ipynb
rows_train = sum(len(row) for row in pd.read_csv(r"Temp_Data\train_data.csv",chunksize=50000))
rows_test = sum(len(row) for row in pd.read_csv(r"Temp_Data\test_data.csv",chunksize=50000))


# %%
# pradedame mokinti ann
ann.fit_generator(
    generator(r"Temp_Data\train_data.csv", batch_size),
    epochs=2,
    steps_per_epoch = rows_train/batch_size,
    validation_data = generator(r"Temp_Data\test_data.csv", batch_size),
    validation_steps = rows_test/batch_size
)


