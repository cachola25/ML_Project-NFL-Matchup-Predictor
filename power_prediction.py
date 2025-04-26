#!/usr/bin/env python
# coding: utf-8

# In[12]:


import re
import os
import glob
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import Model



# In[3]:


def sort_dirs(path):
    path = os.path.basename(path)
    if path.startswith("."):
        return 0
    found = re.search(r'\((\d+)\)', path)
    if not found:
        return 0
    return int(found.group(1))

def sort_prod(file):
    if file.startswith("."):
        return 0
    try:
        found = re.search(r'2022-\d{2}-\d{2}', file)
        index = found.end()
    except Exception as e:
        print(f"ERR on {file}")
    return int(file[index - 2:file.index(".csv")])

def split_time_day(df):
    df[['time','day']] = (
        df['time_day']
           .str.split('/', expand=True)
           .apply(lambda col: col.str.strip())
    )
    df['day'] = df['day'].astype(int)
    df.drop('time_day', axis=1, inplace=True)
    return df

def parse_conditions(dir):
    path = os.path.join(os.getcwd(), dir)
    # Get diagram 1 data into a dataframe
    full_path = os.path.join(path, "Weather_Diagram_1*")
    diagram_1_data = []
    for file in sorted(glob.glob(full_path), key=sort_dirs):
        if file.startswith("."):
            continue
        data = pd.read_csv(file, sep=";",skiprows=1,header=None,
                           names=["time_day","ambient","module_temp","wind"])
        diagram_1_data.append(data)
    df1 = pd.concat(diagram_1_data, ignore_index=True)
    df1 = split_time_day(df1)

    # Get diagram 2 data into a dataframe
    full_path = os.path.join(path, "Weather_Diagram_2*")
    diagram_2_data = []
    for file in sorted(glob.glob(full_path), key=sort_dirs):
        if file.startswith("."):
            continue
        data = pd.read_csv(file, sep=";",skiprows=1,header=None,
                           names=["time_day","insolation"])
        diagram_2_data.append(data)
    df2 = pd.concat(diagram_2_data, ignore_index=True)
    df2 = split_time_day(df2)

    # Merge into one dataframe
    combined_df = pd.merge(df1,df2,on=['day','time'])
    return combined_df

def parse_production(dir):
    path = os.path.join(os.getcwd(), dir)
    df = []
    for file in sorted(os.listdir(path),key=sort_prod):
        if file.startswith("."):
            continue
        full_path = f"{path}/{file}"
        data = pd.read_csv(full_path, sep=";",skiprows=1,header=None, 
                           names=["time","power"])
        day = int(file.split("-")[-1].split(".")[0])
        data['day'] = day
        df.append(data)
    df = pd.concat(df, ignore_index=True)
    df['time'] = df['time'].str.strip()
    df['power'] = pd.to_numeric(df['power'], errors='coerce')
    return df

# Load data
data_dir = "PVSystem/"

all_inputs = []
all_outputs = []
by_month_data = dict()
for dir in sorted(os.listdir(data_dir)):
    full_path = os.path.join(data_dir, dir)
    if not os.path.isdir(full_path) or dir.startswith("."):
        continue
    cond_df = parse_conditions(full_path + '/Conditions')
    prod_df = parse_production(full_path + '/Production')
    all_inputs.append(cond_df.assign(month=dir))
    all_outputs.append(prod_df.assign(month=dir))

    by_month_data[dir] = dict()
    by_month_data[dir]['cond'] = []
    by_month_data[dir]['prod'] = []
    by_month_data[dir]['cond'].append(cond_df.assign(month=dir))
    by_month_data[dir]['prod'].append(prod_df.assign(month=dir))

all_inputs = pd.concat(all_inputs, ignore_index=True)
all_outputs = pd.concat(all_outputs, ignore_index=True)
all_data = pd.merge(
    all_inputs,
    all_outputs,
    on=['month','day','time'],
    how='left'
)


# In[ ]:


# Create and train model
all_data['hour'] = all_data['time'].str.extract(r'(\d+):').astype(int)
init_features = ['ambient', 'wind', 'insolation', 'month', 'day', 'hour']
mod_temp_feature = ['module_temp']
target = ['power']

data = all_data[init_features + mod_temp_feature + target].dropna().reset_index(drop=True)

init_scaler = MinMaxScaler()
mod_temp_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = init_scaler.fit_transform(data[init_features])
mod_temp_scaled = mod_temp_scaler.fit_transform(data[mod_temp_feature])
Y_scaled = target_scaler.fit_transform(data[target])

X_train, X_val = [], []
mod_temp_train, mod_temp_val = [], []
Y_train, Y_val = [], []

SEQ_LEN = 12
def create_sequences(month_df):
    X  = init_scaler.transform(month_df[init_features])
    mod_temp  = mod_temp_scaler.transform(month_df[mod_temp_feature])
    Y  = target_scaler.transform(month_df[target])
    Xs, Y1s, Y2s = [], [], []
    for i in range(len(X) - SEQ_LEN):
        Xs.append(X[i:i+SEQ_LEN])
        Y1s.append(mod_temp[i+SEQ_LEN])
        Y2s.append(Y[i+SEQ_LEN])
    return np.array(Xs), np.array(Y1s), np.array(Y2s)

for month in sorted(by_month_data):
    cond_df = by_month_data[month]["cond"][0]
    prod_df = by_month_data[month]["prod"][0]
    temp = cond_df.merge(prod_df, on=["month", "day", "time"], how="left")
    temp["hour"] = temp["time"].str.extract(r'(\d+):').astype(int)

    month_df = temp[init_features + mod_temp_feature + target].dropna().reset_index(drop=True)

    x, mod_temp, y = create_sequences(month_df)
    cutoff = int(0.8 * len(x))

    X_train.append(x[:cutoff])
    X_val.append(x[cutoff:])
    mod_temp_train.append(mod_temp[:cutoff])
    mod_temp_val.append(mod_temp[cutoff:])
    Y_train.append(y[:cutoff])
    Y_val.append(y[cutoff:])

X_train = np.concatenate(X_train)
X_val = np.concatenate(X_val)
mod_temp_train = np.concatenate(mod_temp_train)
mod_temp_val = np.concatenate(mod_temp_val)
Y_train = np.concatenate(Y_train)
Y_val = np.concatenate(Y_val)
print("Train shape :", X_train.shape, Y_train.shape)
print("Val   shape :", X_val.shape,   Y_val.shape)


# In[16]:


tf.random.set_seed(42)

# Define model architecture
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

# Layers that predict are used before mod_temp and power predictions
shared_layers = LSTM(64, recurrent_activation='sigmoid',activation='tanh',return_sequences=True)(input_layer)
shared_layers = Dropout(0.2)(shared_layers)
shared_layers = LSTM(32)(shared_layers)

# Layers to predict mod_temp
mod_temp_layers = Dense(16, activation='relu')(shared_layers)
mod_temp_layers = Dense(1, name="module_temp")(mod_temp_layers)

# Layers to predict power
power_layers = Concatenate()([shared_layers, mod_temp_layers])
power_layers = Dense(32, activation='relu')(power_layers)
power_layers = Dense(1, name="power")(power_layers)

model = Model(inputs=input_layer, outputs=[mod_temp_layers, power_layers])
model.compile(
    optimizer="adam",
    loss={"module_temp": "mse", "power": "mse"},
    loss_weights={"module_temp": 0.3, "power": 1.0},
)
model.summary()
# model_name = "power_prediction_model.keras"
# if not os.path.isfile(os.path.join(os.getcwd(),model_name)):
#     model = Sequential([
#         Input(shape=(SEQ_LEN, X_seq.shape[2])),
#         LSTM(64, recurrent_activation='sigmoid',activation='tanh',return_sequences=True),
#         Dropout(0.2),
#         LSTM(32, recurrent_activation='sigmoid',activation='tanh'),
#         Dense(32, activation='relu'),
#         Dense(1)
#     ])

#     model.compile(optimizer='adam', 
#                   loss='mse',
#                   metrics=['mse']        
#     )
#     log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#     tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
#     earlystopping_callback = EarlyStopping(patience=20, restore_best_weights=True)

#     history = model.fit(
#         X_train, Y_train,
#         validation_data=(X_val, Y_val),
#         epochs=500,
#         batch_size=64,
#         callbacks=[earlystopping_callback, tensorboard_callback],
#     )

#     model.evaluate(X_val, Y_val)
#     # model.save("power_prediction_model.keras")


# In[ ]:


log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
earlystopping_callback = EarlyStopping(patience=20, restore_best_weights=True)

history = model.fit(
    X_train,
    {"module_temp": mod_temp_train, "power": Y_train},
    validation_data=(X_val,
                     {"module_temp": mod_temp_val, "power": Y_val}),
    epochs=300,
    batch_size=64,
    callbacks=[tensorboard_callback, earlystopping_callback],
    verbose=1,
)


# In[4]:


avg = []
for xi,xp, y in zip(X_val, model(X_val), Y_val):
    err = (1 - (abs((xp - y)))) * 100
    avg.append(err)
print(f"Avg acc: {np.mean(avg)}")


# In[11]:


month = {
    "01": "Jan",
    "02": "Feb",
    "03": "Mar",
    "04": "Apr",
    "05": "May",
    "06": "Jun",
    "07": "Jul",
    "08": "Aug",
    "09": "Sep",
    "10": "Oct",
    "11": "Nov",
    "12": "Dec"
}

num_months = len(by_month_data)
cols = 3
rows = math.ceil(num_months / cols)
fig, axs = plt.subplots(rows, cols, figsize=(18, 5 * rows))
plt.subplots_adjust(hspace=0.5)
axs = axs.flatten()
idx = 0

for key in by_month_data:
    input_df, output_df = by_month_data[key]['cond'][0], by_month_data[key]['prod'][0]
    all_data = pd.merge(input_df, output_df, on=['month','day','time'], how='left')
    all_data['hour']   = all_data['time'].str.extract(r'(\d+):').astype(int)

    data = all_data[features + target].dropna().reset_index(drop=True)
    X_scaled = scaler_X.transform(data[features])
    Y_scaled = scaler_Y.transform(data[target])

    X_seq, Y_seq = create_sequences(X_scaled, Y_scaled, SEQ_LEN)
    Y_pred_scaled = model(X_seq)
    pred_unscaled = scaler_Y.inverse_transform(Y_pred_scaled).flatten()
    true_unscaled = scaler_Y.inverse_transform(Y_seq).flatten()

    ax = axs[idx]
    ax.plot(pred_unscaled, label="Predicted")
    ax.plot(true_unscaled, label="True")
    ax.set_title(f"Month: {month[key]}")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Power (kW)")
    ax.legend()

    idx += 1

for j in range(idx, len(axs)):
    axs[j].axis('off')

plt.show()

