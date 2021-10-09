# --------------------------------------------------
#
#     Copyright (C) 2021 Kevin Bronik
#     UCL Department of Mathematics
#     UCL EGA Institute for Women's Health

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#
#     {Machine learning algorithms to analyze serial multi-dimensional data to predict prostate
#     cancer progression}
#     This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
#     This is free software, and you are welcome to redistribute it
#     under certain conditions; type `show c' for details.
from keras.layers import Dense, Dropout, Flatten, Input, UpSampling3D, Lambda, Reshape
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.advanced_activations import PReLU as prelu
from keras.layers.normalization import BatchNormalization as BN
import tensorflow as tf
from keras import backend as K
import numpy as np
from keras import regularizers
from keras.models import Model
# K.set_image_dim_ordering('th')
import keras
K.set_image_data_format('channels_last')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping,  TensorBoard, LambdaCallback, ModelCheckpoint
# from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import os
import shutil
import sys
import signal
import subprocess
import time

# import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import pad_sequences

CEND      = '\33[0m'
CBOLD     = '\33[1m'
CITALIC   = '\33[3m'
CURL      = '\33[4m'
CBLINK    = '\33[5m'
CBLINK2   = '\33[6m'
CSELECTED = '\33[7m'

CBLACK  = '\33[30m'
CRED    = '\33[31m'
CGREEN  = '\33[32m'
CYELLOW = '\33[33m'
CBLUE   = '\33[34m'
CVIOLET = '\33[35m'
CBEIGE  = '\33[36m'
CWHITE  = '\33[37m'

CBLACKBG  = '\33[40m'
CREDBG    = '\33[41m'
CGREENBG  = '\33[42m'
CYELLOWBG = '\33[43m'
CBLUEBG   = '\33[44m'
CVIOLETBG = '\33[45m'
CBEIGEBG  = '\33[46m'
CWHITEBG  = '\33[47m'

CGREY    = '\33[90m'
CRED2    = '\33[91m'
CGREEN2  = '\33[92m'
CYELLOW2 = '\33[93m'
CBLUE2   = '\33[94m'
CVIOLET2 = '\33[95m'
CBEIGE2  = '\33[96m'
CWHITE2  = '\33[97m'

CGREYBG    = '\33[100m'
CREDBG2    = '\33[101m'
CGREENBG2  = '\33[102m'
CYELLOWBG2 = '\33[103m'
CBLUEBG2   = '\33[104m'
CVIOLETBG2 = '\33[105m'
CBEIGEBG2  = '\33[106m'
CWHITEBG2  = '\33[107m'

batch_size = 1
input_dim = 28
units = 4
output_size = 2  # labels are from 0 to 9

def data_loading(arg1=None, arg2=None, arg31=None, arg32=None, arg33=None, arg34=None, arg35=None, arg36=None, feat=None):
    global dfx, ytrain, i, j
    dfx = pd.read_csv('Serial radiomics database send out_280921.csv')
    # print(dfx)

    ytrain = dfx["Histopathological progression (0=no, 1=yes)"]
    ytrain = np.array(ytrain)

    dfx_adc0_label = pd.read_csv(arg1)
    PatientName0_label = dfx_adc0_label["PatientName"]
    PatientName0_label = np.array(PatientName0_label)
    # print('PatientName', PatientName0_label)
    # dfx["Date off AS/still on AS"] = dfx["Date off AS/still on AS"].fillna(0)
    dfx[arg2] = dfx[arg2].fillna(0)
    label0_date = pd.to_datetime(dfx[arg2], dayfirst=True)

    dfx_adc1 = pd.read_csv(arg31)
    MRI1 = dfx_adc1[feat]
    MRI1 = np.array(MRI1)
    PatientName1 = dfx_adc1["PatientName"]
    PatientName1 = np.array(PatientName1)

    dfx["Date of MRI (1)"] = dfx["Date of MRI (1)"].fillna(0)
    MRI1_date = pd.to_datetime(dfx["Date of MRI (1)"], dayfirst=True)
    # /// 2
    # dfx_adc2 = pd.read_csv('RobustRadiomics_ADC_Scan2_serial.csv')
    dfx_adc2 = pd.read_csv(arg32)
    MRI2 = dfx_adc2[feat]
    MRI2 = np.array(MRI2)
    PatientName2 = dfx_adc2["PatientName"]
    PatientName2 = np.array(PatientName2)

    dfx["Date of MRI (2)"] = dfx["Date of MRI (2)"].fillna(0)
    MRI2_date = pd.to_datetime(dfx["Date of MRI (2)"], dayfirst=True)

    dfx_adc3 = pd.read_csv(arg33)
    MRI3 = dfx_adc3[feat]
    MRI3 = np.array(MRI3)
    PatientName3 = dfx_adc3["PatientName"]
    PatientName3 = np.array(PatientName3)
    # print('PatientName3', PatientName3)

    dfx["Date of MRI (3)"] = dfx["Date of MRI (3)"].fillna(0)
    MRI3_date = pd.to_datetime(dfx["Date of MRI (3)"], dayfirst=True)
    # /// 4
    # dfx_adc4 = pd.read_csv('RobustRadiomics_ADC_Scan4_serial.csv')
    dfx_adc4 = pd.read_csv(arg34)
    MRI4 = dfx_adc4[feat]
    MRI4 = np.array(MRI4)
    PatientName4 = dfx_adc4["PatientName"]
    PatientName4 = np.array(PatientName4)
    # print('PatientName4', PatientName4)

    dfx["Date of MRI (4)"] = dfx["Date of MRI (4)"].fillna(0)
    MRI4_date = pd.to_datetime(dfx["Date of MRI (4)"], dayfirst=True)
    # /// 5
    # dfx_adc5 = pd.read_csv('RobustRadiomics_ADC_Scan5_serial.csv')
    dfx_adc5 = pd.read_csv(arg35)
    MRI5 = dfx_adc5[feat]
    MRI5 = np.array(MRI5)
    PatientName5 = dfx_adc5["PatientName"]
    PatientName5 = np.array(PatientName5)
    # print('PatientName5', PatientName5)
    # print('MRI5:', MRI5.shape)
    dfx["Date of MRI (5)"] = dfx["Date of MRI (5)"].fillna(0)
    MRI5_date = pd.to_datetime(dfx["Date of MRI (5)"], dayfirst=True)
    # /// 6 final
    # dfx_adc6 = pd.read_csv('RobustRadiomics_ADC_final_serial.csv')
    dfx_adc6 = pd.read_csv(arg36)
    MRI6 = dfx_adc6[feat]
    MRI6 = np.array(MRI6)
    PatientName6 = dfx_adc6["PatientName"]
    PatientName6 = np.array(PatientName6)
    # print('PatientName6', PatientName6)

    dfx["Date of MRI (6)"] = dfx["Date of MRI (6)"].fillna(0)
    MRI6_date = pd.to_datetime(dfx["Date of MRI (6)"], dayfirst=True)
    MR_D1 = []
    MR_D2 = []
    MR_D3 = []
    MR_D4 = []
    MR_D5 = []
    MR_D6 = []
    for i in range(len(label0_date)):
        MR_D1.append(pd.Timedelta(pd.to_datetime(label0_date[i]) - pd.to_datetime(MRI1_date[i])).days / 365)
        MR_D2.append(pd.Timedelta(pd.to_datetime(label0_date[i]) - pd.to_datetime(MRI2_date[i])).days / 365)
        MR_D3.append(pd.Timedelta(pd.to_datetime(label0_date[i]) - pd.to_datetime(MRI3_date[i])).days / 365)
        MR_D4.append(pd.Timedelta(pd.to_datetime(label0_date[i]) - pd.to_datetime(MRI4_date[i])).days / 365)
        MR_D5.append(pd.Timedelta(pd.to_datetime(label0_date[i]) - pd.to_datetime(MRI5_date[i])).days / 365)
        MR_D6.append(pd.Timedelta(pd.to_datetime(label0_date[i]) - pd.to_datetime(MRI6_date[i])).days / 365)

    testY = {}
    testX = {}
    for pn in PatientName0_label:
        r = np.where(PatientName0_label == pn)
        i = r[0][0]
        tmp1 = []
        tmp2 = []
        tmp1.append(MR_D6[i])
        tmp1.append(MR_D5[i])
        tmp1.append(MR_D4[i])
        tmp1.append(MR_D3[i])
        tmp1.append(MR_D2[i])
        tmp1.append(MR_D1[i])
        if pn in PatientName1:
            r = np.where(PatientName1 == pn)
            j = r[0][0]
            tmp2.append(MRI1[j])
        if pn in PatientName2:
            r = np.where(PatientName2 == pn)
            j = r[0][0]
            tmp2.append(MRI2[j])
        if pn in PatientName3:
            r = np.where(PatientName3 == pn)
            j = r[0][0]
            tmp2.append(MRI3[j])
        if pn in PatientName4:
            r = np.where(PatientName4 == pn)
            j = r[0][0]
            tmp2.append(MRI4[j])
        if pn in PatientName5:
            r = np.where(PatientName5 == pn)
            j = r[0][0]
            tmp2.append(MRI5[j])
        if pn in PatientName6:
            r = np.where(PatientName6 == pn)
            j = r[0][0]
            tmp2.append(MRI6[j])
        testY[i] = tmp1
        testX[i] = tmp2

    return testX

# dfx_adc0_label = pd.read_csv('RobustRadiomics_ADC_Final_serial.csv')
in1 = 'RobustRadiomics_ADC_Final_serial.csv'
# dfx["Date off AS/still on AS"] = dfx["Date off AS/still on AS"].fillna(0)
feat1 = "Date off AS/still on AS"
# dfx_adc1 = pd.read_csv('RobustRadiomics_ADC_Scan1_serial.csv')
file1 = 'RobustRadiomics_ADC_Scan1_serial.csv'
# dfx_adc1 = pd.read_csv('RobustRadiomics_ADC_Scan2_serial.csv')
file2 = 'RobustRadiomics_ADC_Scan2_serial.csv'
# dfx_adc1 = pd.read_csv('RobustRadiomics_ADC_Scan3_serial.csv')
file3 = 'RobustRadiomics_ADC_Scan3_serial.csv'
# dfx_adc1 = pd.read_csv('RobustRadiomics_ADC_Scan4_serial.csv')
file4 = 'RobustRadiomics_ADC_Scan4_serial.csv'
# dfx_adc1 = pd.read_csv('RobustRadiomics_ADC_Scan5_serial.csv')
file5 = 'RobustRadiomics_ADC_Scan5_serial.csv'
# dfx_adc6 = pd.read_csv('RobustRadiomics_ADC_final_serial.csv')
final = 'RobustRadiomics_ADC_final_serial.csv'
feat2 = "original_firstorder_10Percentile"
df_final = pd.read_csv('RobustRadiomics_ADC_final_serial.csv')


column_names = []
for col in df_final.columns:
    column_names.append(col)
# print('header', header)
column_names.remove('PatientName')
# print('column_names', column_names)
T1_complete = []
i = 0
for cn in column_names:
    T1_complete.append(data_loading(arg1=in1, arg2=feat1, arg31=file1, arg32=file2, arg33=file3, arg34=file4, arg35=file5,
                       arg36=final, feat=cn))
    i = i + 1
TRAIN1 = []

for k in range(len(T1_complete[0])):
    pa1_feat = []
    for j in range(len(T1_complete[0][k])):
        tmp1x = []
        for i in range(len(column_names)):
            tmp1x.append(T1_complete[i][k][j].astype(dtype=np.float32))


        pa1_feat.append(tmp1x)
    TRAIN1.append(pa1_feat)


in1x = 'RobustRadiomics_T2w_Final_serial.csv'
# dfx["Date off AS/still on AS"] = dfx["Date off AS/still on AS"].fillna(0)
feat1x = "Date off AS/still on AS"
# dfx_adc1 = pd.read_csv('RobustRadiomics_ADC_Scan1_serial.csv')
file1x = 'RobustRadiomics_T2w_Scan1_serial.csv'
# dfx_adc1 = pd.read_csv('RobustRadiomics_ADC_Scan2_serial.csv')
file2x = 'RobustRadiomics_T2w_Scan2_serial.csv'
# dfx_adc1 = pd.read_csv('RobustRadiomics_ADC_Scan3_serial.csv')
file3x = 'RobustRadiomics_T2w_Scan3_serial.csv'
# dfx_adc1 = pd.read_csv('RobustRadiomics_ADC_Scan4_serial.csv')
file4x = 'RobustRadiomics_T2w_Scan4_serial.csv'
# dfx_adc1 = pd.read_csv('RobustRadiomics_ADC_Scan5_serial.csv')
file5x = 'RobustRadiomics_T2w_Scan5_serial.csv'
# dfx_adc6 = pd.read_csv('RobustRadiomics_ADC_final_serial.csv')
finalx = 'RobustRadiomics_T2w_Final_serial.csv'
feat2x = "original_firstorder_10Percentile"
df_finalx = pd.read_csv('RobustRadiomics_T2w_Final_serial.csv')


column_namesx = []
for col in df_finalx.columns:
    column_namesx.append(col)
# print('header', header)
column_namesx.remove('PatientName')
# print('column_namesx', column_namesx)
T2_complete = []
i = 0
for cn in column_namesx:
    T2_complete.append(data_loading(arg1=in1x, arg2=feat1x, arg31=file1x, arg32=file2x, arg33=file3x, arg34=file4x, arg35=file5x,
                                    arg36=finalx, feat=cn))
    i = i + 1
TRAIN2 = []

for k in range(len(T2_complete[0])):
    pa2_feat = []
    for j in range(len(T2_complete[0][k])):
        tmp2x = []
        for i in range(len(column_namesx)):
            tmp2x.append(T2_complete[i][k][j].astype(dtype=np.float32))


        pa2_feat.append(tmp2x)
    TRAIN2.append(pa2_feat)

dfy = pd.read_csv('Serial radiomics database send out_280921.csv')
Ytrain = dfy["Histopathological progression (0=no, 1=yes)"]
Ytrain = np.array(Ytrain)
TRAIN1 = np.array(TRAIN1)
TRAIN2 = np.array(TRAIN2)
# print('Label_shape', Ytrain.shape)
# print('TRAIN1_shape(inputX1)', TRAIN1.shape)
# print('TRAIN2_shape(inputX2)', TRAIN2.shape)

print('\x1b[6;30;41m' + "                                                              " + '\x1b[0m')
print('\x1b[6;30;41m' + "Loading data into memory is done, training begins ...         " + '\x1b[0m')
print('\x1b[6;30;41m' + "                                                              " + '\x1b[0m')


def build_model_GRU(allow_cudnn_kernel=True):

    in1 = keras.layers.Input(shape=(None, 27), name='in1',  ragged=True)
    in2 = keras.layers.Input(shape=(None, 17), name='in2',  ragged=True)
    # in1 = keras.layers.Input(shape=(None, None), name='in1')
    # in2 = keras.layers.Input(shape=(None, None), name='in2')
    # print(in1)

    # lstm_layer1 = keras.layers.RNN(keras.layers.LSTMCell(units), return_sequences=False)(in1)
    # lstm_layer2 = keras.layers.RNN(keras.layers.LSTMCell(units), return_sequences=False)(in2)  32
    lstm_layer1 = keras.layers.GRU(200, return_sequences=True)(in1)
    lstm_layer2 = keras.layers.GRU(200, return_sequences=True)(in2)

    merged = keras.layers.concatenate([lstm_layer1, lstm_layer2], axis=-1)
    # 64
    layer = keras.layers.GRU(400, return_sequences=True,  dropout=0.5)(merged)
    # layer = keras.layers.GRU(64, return_sequences=True)(layer)
    # 32
    layer = keras.layers.GRU(200, return_sequences=True,  dropout=0.5)(layer)
    # layer = keras.layers.GRU(32, return_sequences=True)(layer)
    # 16
    layer = keras.layers.GRU(100, return_sequences=True, dropout=0.5)(layer)
    layer = keras.layers.GRU(50, return_sequences=True,  dropout=0.5)(layer)
    layer = keras.layers.GRU(40, return_sequences=True,  dropout=0.5)(layer)
    layer = keras.layers.GRU(30, return_sequences=True,  dropout=0.5)(layer)
    layer = keras.layers.GRU(20, return_sequences=False)(layer)
    layer = keras.layers.BatchNormalization()(layer)

    # output = keras.layers.Dense(output_size, activation='sigmoid')(layer)
    layer = keras.layers.Dense(20, activation=None)(layer)
    layer = prelu(name='prelu_1')(layer)
    layer = keras.layers.Dropout(0.5)(layer)
    layer = keras.layers.Dense(10, activation=None)(layer)
    layer = prelu(name='prelu_2')(layer)
    layer = keras.layers.Dropout(0.5)(layer)
    layer = keras.layers.Dense(5, activation=None)(layer)
    layer = prelu(name='prelu_3')(layer)
    # layer = keras.layers.Dropout(0.2)(layer)
    # output = keras.layers.Dense(output_size)(layer)
    output = Dense(units=2, name='out', activation='softmax')(layer)
    model = Model(inputs=[in1, in2], outputs=output)
    # model = Model(inputs=in1, outputs=output)
    return model


def normalize_data(im, datatype=np.float32):
    im1 = im.astype(dtype=datatype)
    im2 = im.max()
    normalized = im1 / im2
    return normalized

def _to_tensor(x, dtype=tf.float32):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x
# print('TRAIN1',type(TRAIN1))

# unbalanced_training_and_testing
x_train_1 = TRAIN1[:50,]
x_train_2 = TRAIN2[:50,]
x_test_1 = TRAIN1[50:66,]
x_test_2 = TRAIN2[50:66,]
sample_1 = TRAIN1[66:,]
sample_2 = TRAIN2[66:,]
y_train = Ytrain[:50]
y_test = Ytrain[50:66]
sample_label = Ytrain[66:]

x_train_1 = tf.ragged.constant(x_train_1)
x_train_2 = tf.ragged.constant(x_train_2)
x_test_1 = tf.ragged.constant(x_test_1)
x_test_2 = tf.ragged.constant(x_test_2)
sample_1 = tf.ragged.constant(sample_1)
sample_2 = tf.ragged.constant(sample_2)

def accuracy_loss(y_true, y_pred):
    a_list = tf.convert_to_tensor(y_pred)
    # print('a_list ', a_list)

    # y_pred = tf.stack(a_list)
    a_list = tf.where(tf.less_equal(a_list, 0.5), 0.0, 1.0)
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - a_list
    fp = K.sum(neg_y_true * a_list)
    tn = K.sum(neg_y_true * neg_y_pred)
    fn = K.sum(y_true * neg_y_pred)
    tp = K.sum(y_true * a_list)
    acc = (tp + tn) / (tp + tn + fn + fp)
    return 1.0 - acc


def specificity(y_true, y_pred):
    a_list = tf.convert_to_tensor(y_pred)
    # print('a_list ', a_list)

    # y_pred = tf.stack(a_list)
    a_list = tf.where(tf.less_equal(a_list, 0.5), 0.0, 1.0)
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - a_list
    fp = K.sum(neg_y_true * a_list)
    tn = K.sum(neg_y_true * neg_y_pred)
    spec = tn / (tn + fp + K.epsilon())
    return spec


def specificity_loss(y_true, y_pred):
    return 1.0 - specificity(y_true, y_pred)


def sensitivity(y_true, y_pred):
    # neg_y_true = 1 - y_true
    a_list = tf.convert_to_tensor(y_pred)
    # print('a_list ', a_list)

    a_list = tf.where(tf.less_equal(a_list, 0.5), 0.0, 1.0)
    neg_y_pred = 1 - a_list
    fn = K.sum(y_true * neg_y_pred)
    tp = K.sum(y_true * a_list)
    sens = tp / (tp + fn + K.epsilon())
    return sens

def sensitivity_loss(y_true, y_pred):
    return 1.0 - sensitivity(y_true, y_pred)

def precision(y_true, y_pred):
    a_list = tf.convert_to_tensor(y_pred)
    # print('a_list ', a_list)
    # a_list[:, 0][a_list[:,0] > 0.5] = 1.0
    # a_list[:, 1][a_list[:,1] > 0.5] = 1.0
    # a_list[:, 0][a_list[:,0] < 0.5] = 0.0
    # a_list[:, 1][a_list[:,1] < 0.5] = 0.0
    # y_pred = tf.stack(a_list)
    a_list = tf.where(tf.less_equal(a_list, 0.5), 0.0, 1.0)
    neg_y_true = 1 - y_true
    # neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * a_list)
    tp = K.sum(y_true * a_list)
    pres = tp / (tp + fp + K.epsilon())
    return pres

def precision_loss(y_true, y_pred):
    return 1.0 - precision(y_true, y_pred)

def Mattews_loss(y_true, y_pred):
    a_list = tf.convert_to_tensor(y_pred)
    # print('a_list ', a_list)
    a_list = tf.where(tf.less_equal(a_list, 0.5), 0.0, 1.0)

    neg_y_true = 1 - y_true
    neg_y_pred = 1 - a_list

    tp = K.sum(y_true * a_list)
    fn = K.sum(y_true * neg_y_pred)


    fp = K.sum(neg_y_true * a_list)
    tn = K.sum(neg_y_true * neg_y_pred)
    mult1 = tp * tn
    mult2 = fp * fn

    sum1 = fp + tp
    sum2 = tp + fn
    sum3 = tn + fp
    sum4 = tn + fn

    mcc = (mult1 - mult2) / tf.math.sqrt(sum1 * sum2 * sum3 * sum4)

    return K.abs(1 - mcc)
def concatenated_loss(y_true, y_pred):

    loss = keras.losses.binary_crossentropy(y_true, y_pred) + accuracy_loss(y_true, y_pred) + \
           sensitivity_loss(y_true, y_pred)  \
           + precision_loss(y_true, y_pred) + specificity_loss(y_true, y_pred)
    return loss

    # return keras.losses.binary_crossentropy(y_true, y_pred)
# model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
model = build_model_GRU(allow_cudnn_kernel=True)
opt = tf.keras.optimizers.Adam(lr=1e-5)
optimizer = tf.keras.optimizers.Adam(lr=1e-8, clipvalue=0.5, clipnorm=0.5)

model.compile(
    # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # loss=keras.losses.SparseCategoricalCrossentropy(),
    loss=concatenated_loss,
    # optimizer="sgd",
    # optimizer='adadelta',
    optimizer=opt,
    metrics=["accuracy", precision, sensitivity, specificity],
    # metrics=["accuracy"],
)
model.summary()
# from keras.utils.vis_utils import plot_model
# tf.keras.utils.plot_model(model, to_file='model_plot1.png', show_shapes=True, show_layer_names=True)

def predict_test_prob(model):
    all_predictions_prob = []
    with tf.device("CPU:0"):
        # cpu_model = build_model(allow_cudnn_kernel=True)
        model.set_weights(model.get_weights())
        for i in range(len(sample_label)):
            t1 = tf.expand_dims(sample_1[i], 0)
            t2 = tf.expand_dims(sample_2[i], 0)
            # result = tf.argmax(cpu_model.predict_on_batch(tf.expand_dims(sample[i], 0)), axis=1)
            result = model.predict_on_batch([t1, t2])
            # all_predictions_prob.append(np.max(np.array(result)))
            all_predictions_prob.append(np.array(result)[0][1])
            # result = cpu_model.predict_on_batch([t1, t2])
            print(
                "Predicted result is: %s, target result is: %s" % (np.array(tf.argmax(result, axis=1)), sample_label[i])
                # "Predicted result is:"'\x1b[6;30;42m' + "%s" + '\x1b[0m', "target result is: %s" % (result.numpy(), sample_label)

            )
        # plt.imshow(sample, cmap=plt.get_cmap("gray"))
        return all_predictions_prob

def run_test(model):
    predicted_results = predict_test_prob(model)
    print('predicted_results:', predicted_results)
    # sample_label = np.array(sample_label)
    print('sample_label:', sample_label)
    ns_probs = [0 for _ in range(len(sample_label))]
    print('ns_probs', ns_probs)
    ns_auc = roc_auc_score(sample_label, ns_probs)
    lr_auc = roc_auc_score(sample_label, predicted_results)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Predicted: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(sample_label, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(sample_label, predicted_results)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Predicted')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

def plot_callback(model):
    all_predictions = []
    with tf.device("CPU:0"):
        # cpu_model = build_model(allow_cudnn_kernel=True)
        model.set_weights(model.get_weights())
        for i in range(len(sample_label)):
            t1 = tf.expand_dims(sample_1[i], 0)
            t2 = tf.expand_dims(sample_2[i], 0)
            # result = tf.argmax(cpu_model.predict_on_batch(tf.expand_dims(sample[i], 0)), axis=1)
            result = model.predict_on_batch([t1, t2])
            result1 = tf.argmax(result, axis=1)
            all_predictions.append(np.array(result1)[0])
            # result = cpu_model.predict_on_batch([t1, t2])
            print(
                "Predicted result [probably(N)  probably(P)] is: %s, Predicted result argmax: is: %s, Target result is: %s" %(np.array(result), np.array(result1), sample_label[i])
                # "Predicted result is:"'\x1b[6;30;42m' + "%s" + '\x1b[0m', "target result is: %s" % (result.numpy(), sample_label)

            )

        all_predictions = np.array(all_predictions)
        # print('all_predictions:', all_predictions)
        neg_y_true = 1 - sample_label
        # neg_y_pred = 1 - y_pred
        neg_y_pred = 1 - all_predictions
        tn = np.sum(neg_y_true * neg_y_pred)
        fn = np.sum(sample_label * neg_y_pred)
        fp = np.sum(neg_y_true * all_predictions)
        tp = np.sum(sample_label * all_predictions)
        print('False positive: ', CRED + str(fp) + CEND)
        print('True positive:  ', CRED + str(tp) + CEND)
        print('False negative: ', CRED + str(fn) + CEND)
        print('True negative:  ', CRED + str(tn) + CEND)
        pres = tp / (tp + fp + 1.1920929e-07)
        sens = tp / (tp + fn + 1.1920929e-07)
        spec = tn / (tn + fp + 1.1920929e-07)

        print(CYELLOW + "Testing Precision:   ", CRED + str(pres) + CEND)
        print(CYELLOW + "Testing Sensitivity: ", CRED + str(sens) + CEND)
        print(CYELLOW + "Testing Specificity: ", CRED + str(spec) + CEND)
        if pres > 0.8 and sens > 0.8 and spec > 0.8:
            print('\x1b[6;30;41m' + "                                                              " + '\x1b[0m')
            print('\x1b[6;30;41m' + "Interrupting training after reaching good performance!        " + '\x1b[0m')
            print('\x1b[6;30;41m' + "                                                              " + '\x1b[0m')
            print(CBLUE + "Precision best value:   ", CRED + str(pres) + CEND)
            print(CBLUE + "Sensitivity best value: ", CRED + str(sens) + CEND)
            print(CBLUE + "Specificity best value: ", CRED + str(spec) + CEND)
            run_test(model)
            sys.stdout.flush()
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)
        # plt.imshow(sample, cmap=plt.get_cmap("gray"))
        plt.close()
settings_patience = 3000
THIS_PATH = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(THIS_PATH, 'MODEL'))
model_weights = THIS_PATH + '/MODEL/' + 'model_weights_GRU_unbalanced.hdf5'
tensorboardlogs = THIS_PATH + '/Tensorboardlogs'
# print('y_train_before:', y_train)
y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)
# print('y_train_shape', y_train.shape)
# print('y_test_shape', y_test.shape)
# print('y_train:', y_train)
# print('y_test:', y_test)
# tf.ragged.constant(x_train_1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=20, verbose=1,
    mode='auto', min_delta=0.001, cooldown=0, min_lr=0)

model.fit(
    [x_train_1,x_train_2], y_train,
    validation_data=([x_test_1, x_test_2], y_test),
    batch_size=batch_size,
    epochs=3000,
    verbose=1,
    callbacks=[reduce_lr, ModelCheckpoint(model_weights,
                           monitor='val_loss',
                           save_best_only=True,
                           save_weights_only=True),
           EarlyStopping(monitor='val_loss',
                         min_delta=0,
                         patience=settings_patience,
                         verbose=0,
                         mode='auto'),
           TensorBoard(log_dir=tensorboardlogs, histogram_freq=0,
                       write_graph=True,  write_images=True),
               LambdaCallback(
                   on_epoch_end=lambda epoch, logs: plot_callback(model)
               )
               ])

def predict_test(model):
    all_predictions = []
    with tf.device("CPU:0"):
        # cpu_model = build_model(allow_cudnn_kernel=True)
        model.set_weights(model.get_weights())
        for i in range(len(sample_label)):
            t1 = tf.expand_dims(sample_1[i], 0)
            t2 = tf.expand_dims(sample_2[i], 0)
            # result = tf.argmax(cpu_model.predict_on_batch(tf.expand_dims(sample[i], 0)), axis=1)
            result = tf.argmax(model.predict_on_batch([t1, t2]), axis=1)
            all_predictions.append(np.array(result)[0])
            # result = cpu_model.predict_on_batch([t1, t2])
            print(
                "Predicted result is: %s, target result is: %s" % (np.array(result), sample_label[i])
                # "Predicted result is:"'\x1b[6;30;42m' + "%s" + '\x1b[0m', "target result is: %s" % (result.numpy(), sample_label)

            )
        all_predictions = np.array(all_predictions)
        return all_predictions

run_test(model)
