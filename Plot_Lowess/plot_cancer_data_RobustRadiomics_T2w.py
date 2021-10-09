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
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
sns.set_style("white")
plt.rc("axes.spines", top=False, right=False)
sns.set_context("paper")

import pandas as pd
import os
import shutil
import sys
import signal
import subprocess
import time

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
THIS_PATH = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(THIS_PATH, 'MODEL'))
plot_save = THIS_PATH + '/T2W_plots/'
def plot_all(feature=None):
    def load_data(input_file1=None, input_file2=None, input_final=None):

        print('Loading data...')
        df1 = pd.read_csv(input_file1)
        df2 = pd.read_csv(input_file2)
        # df3 = pd.read_csv(input_file3)
        # df4 = pd.read_csv(input_file4)
        # df5 = pd.read_csv(input_file5)
        df6 = pd.read_csv(input_final)
        # print('Database length:', len(df1))
        # data1 = df.iloc[0:len(df), 0:28]
        T = []
        for i in range(len(df1)):
            D = []
            data1 = df1.iloc[i, 1:28]
            D.append(data1)
            data2 = df2.iloc[i, 1:28]
            D.append(data2)
            # data3 = df3.iloc[i, 1:28]
            # D.append(data3)
            # data4 = df4.iloc[i, 1:28]
            # D.append(data4)
            # data5 = df5.iloc[i, 1:28]
            # D.append(data5)
            data6 = df6.iloc[i, 1:28]
            D.append(data6)
            T.append(D)

        return np.array(T)

    dfx = pd.read_csv('Serial radiomics database send out_280921.csv')
    # print(dfx)
    ytrain = dfx["Histopathological progression (0=no, 1=yes)"]
    ytrain = np.array(ytrain)
    # print('ytrain:', ytrain.shape)
    # print('ytrain:', ytrain)
    dfx_T2w0_label = pd.read_csv('RobustRadiomics_T2w_Final_serial.csv')
    PatientName0_label = dfx_T2w0_label["PatientName"]
    PatientName0_label = np.array(PatientName0_label)
    # print('PatientName', PatientName0_label)
    dfx["Date off AS/still on AS"] = dfx["Date off AS/still on AS"].fillna(0)
    label0_date = pd.to_datetime(dfx["Date off AS/still on AS"], dayfirst=True)
    # label0_date = np.array(label0_date).astype(int)
    # print('label0_date:', label0_date.shape)
    # print('label0_date:', label0_date)
    # /// 1
    dfx_T2w1 = pd.read_csv('RobustRadiomics_T2w_Scan1_serial.csv')

    MRI1 = dfx_T2w1[feature]
    MRI1 = np.array(MRI1)
    PatientName1 = dfx_T2w1["PatientName"]
    PatientName1 = np.array(PatientName1)
    # print('PatientName2', PatientName1)
    # print('MRI1:', MRI1.shape)
    # print('MRI1:', MRI1)
    # nan is replaced with 19700101
    dfx["Date of MRI (1)"] = dfx["Date of MRI (1)"].fillna(0)
    MRI1_date = pd.to_datetime(dfx["Date of MRI (1)"], dayfirst=True)
    dfx_T2w2 = pd.read_csv('RobustRadiomics_T2w_Scan2_serial.csv')
    MRI2 = dfx_T2w2[feature]
    MRI2 = np.array(MRI2)
    PatientName2 = dfx_T2w2["PatientName"]
    PatientName2 = np.array(PatientName2)
    # print('PatientName2', PatientName2)
    # print('MRI2:', MRI2.shape)
    # print('MRI2:', MRI2)
    # nan is replaced with 19700101
    dfx["Date of MRI (2)"] = dfx["Date of MRI (2)"].fillna(0)
    MRI2_date = pd.to_datetime(dfx["Date of MRI (2)"], dayfirst=True)
    dfx_T2w3 = pd.read_csv('RobustRadiomics_T2w_Scan3_serial.csv')
    MRI3 = dfx_T2w3[feature]
    MRI3 = np.array(MRI3)
    PatientName3 = dfx_T2w3["PatientName"]
    PatientName3 = np.array(PatientName3)
    # print('PatientName3', PatientName3)
    # print('MRI3:', MRI3.shape)
    # print('MRI3:', MRI3)
    # nan is replaced with 19700101
    dfx["Date of MRI (3)"] = dfx["Date of MRI (3)"].fillna(0)
    MRI3_date = pd.to_datetime(dfx["Date of MRI (3)"], dayfirst=True)
    dfx_T2w4 = pd.read_csv('RobustRadiomics_T2w_Scan4_serial.csv')
    MRI4 = dfx_T2w4[feature]
    MRI4 = np.array(MRI4)
    PatientName4 = dfx_T2w4["PatientName"]
    PatientName4 = np.array(PatientName4)
    # print('PatientName4', PatientName4)
    # print('MRI4:', MRI4.shape)
    # print('MRI4:', MRI4)
    # nan is replaced with 19700101
    dfx["Date of MRI (4)"] = dfx["Date of MRI (4)"].fillna(0)
    MRI4_date = pd.to_datetime(dfx["Date of MRI (4)"], dayfirst=True)
    dfx_T2w5 = pd.read_csv('RobustRadiomics_T2w_Scan5_serial.csv')
    MRI5 = dfx_T2w5[feature]
    MRI5 = np.array(MRI5)
    PatientName5 = dfx_T2w5["PatientName"]
    PatientName5 = np.array(PatientName5)
    # print('PatientName5', PatientName5)
    # print('MRI5:', MRI5.shape)
    # print('MRI5:', MRI5)
    # nan is replaced with 19700101
    dfx["Date of MRI (5)"] = dfx["Date of MRI (5)"].fillna(0)
    MRI5_date = pd.to_datetime(dfx["Date of MRI (5)"], dayfirst=True)
    dfx_T2w6 = pd.read_csv('RobustRadiomics_T2w_Final_serial.csv')
    MRI6 = dfx_T2w6[feature]
    MRI6 = np.array(MRI6)
    PatientName6 = dfx_T2w6["PatientName"]
    PatientName6 = np.array(PatientName6)
    # print('PatientName6', PatientName6)
    # print('MRI6:', MRI6.shape)
    # print('MRI6:', MRI6)
    # nan is replaced with 19700101
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
    # print('MR_D1', MR_D1)
    # print('MR_D2', MR_D2)
    # print('MR_D3', MR_D3)
    # print('MR_D4', MR_D4)
    # print('MR_D5', MR_D5)
    # print('MR_D6', MR_D6)
    Tx1 = []
    Ty1 = []
    Dx1 = {}
    Dy1 = {}
    Tx0 = []
    Ty0 = []
    Dx0 = {}
    Dy0 = {}
    testY = {}
    testX = {}
    testY0 = {}
    testX0 = {}
    for pn in PatientName0_label:
        r = np.where(PatientName0_label == pn)
        i = r[0][0]
        if ytrain[i] - 1 == 0:
            tmp1 = []
            tmp2 = []
            tmp1.append(MR_D6[i])
            tmp1.append(MR_D5[i])
            tmp1.append(MR_D4[i])
            tmp1.append(MR_D3[i])
            tmp1.append(MR_D2[i])
            tmp1.append(MR_D1[i])
            if pn in PatientName6:
                r = np.where(PatientName6 == pn)
                j = r[0][0]
                tmp2.append(MRI6[j])
            if pn in PatientName5:
                r = np.where(PatientName5 == pn)
                j = r[0][0]
                tmp2.append(MRI5[j])
            if pn in PatientName4:
                r = np.where(PatientName4 == pn)
                j = r[0][0]
                tmp2.append(MRI4[j])
            if pn in PatientName3:
                r = np.where(PatientName3 == pn)
                j = r[0][0]
                tmp2.append(MRI3[j])
            if pn in PatientName2:
                r = np.where(PatientName2 == pn)
                j = r[0][0]
                tmp2.append(MRI2[j])
            if pn in PatientName1:
                r = np.where(PatientName1 == pn)
                j = r[0][0]
                tmp2.append(MRI1[j])
            testY[i] = tmp1
            testX[i] = tmp2

        else:
            tmp1 = []
            tmp2 = []
            tmp1.append(MR_D6[i])
            tmp1.append(MR_D5[i])
            tmp1.append(MR_D4[i])
            tmp1.append(MR_D3[i])
            tmp1.append(MR_D2[i])
            tmp1.append(MR_D1[i])
            if pn in PatientName6:
                r = np.where(PatientName6 == pn)
                j = r[0][0]
                tmp2.append(MRI6[j])
            if pn in PatientName5:
                r = np.where(PatientName5 == pn)
                j = r[0][0]
                tmp2.append(MRI5[j])
            if pn in PatientName4:
                r = np.where(PatientName4 == pn)
                j = r[0][0]
                tmp2.append(MRI4[j])
            if pn in PatientName3:
                r = np.where(PatientName3 == pn)
                j = r[0][0]
                tmp2.append(MRI3[j])
            if pn in PatientName2:
                r = np.where(PatientName2 == pn)
                j = r[0][0]
                tmp2.append(MRI2[j])
            if pn in PatientName1:
                r = np.where(PatientName1 == pn)
                j = r[0][0]
                tmp2.append(MRI1[j])
            testY0[i] = tmp1
            testX0[i] = tmp2
    cleaned_listY = {}
    Y_arr = []
    X_arr = []
    cleaned_listY0 = {}
    Y_arr0 = []
    X_arr0 = []
    for d in testY:
        cleaned_listY[d] = [x for x in testY[d] if x < 20]
    for d in cleaned_listY:
        for x in cleaned_listY[d]:
            Y_arr.append(x)
    for d in testX:
        for x in testX[d]:
            X_arr.append(x)
    for d in cleaned_listY:

        # for x, y in zip(cleaned_listY[d], testX[d]):
        #     print('x.shape', np.array(cleaned_listY[d]).shape)
        #     print('y.shape', np.array(testX[d]).shape)
        x = np.array(cleaned_listY[d])
        y = np.array(testX[d])
        if x.shape[0] != y.shape[0]:
            r = np.where(cleaned_listY == d)
            # j = r[0][0]
            # print('x------->', x)
            # print('y------->', y)
            # print(testY[d])
            # print(PatientName0_label[d])

            raise ValueError("x and y must have same first dimension, but "
                             "have shapes {} and {}".format(x.shape, y.shape))
    for d in testY0:
        cleaned_listY0[d] = [x for x in testY0[d] if x < 20]
    for d in cleaned_listY0:
        for x in cleaned_listY0[d]:
            Y_arr0.append(x)
    for d in testX0:
        for x in testX0[d]:
            X_arr0.append(x)
    for d in cleaned_listY0:

        x = np.array(cleaned_listY0[d])
        y = np.array(testX0[d])
        if x.shape[0] != y.shape[0]:
            r = np.where(cleaned_listY0 == d)
            # j = r[0][0]
            # print('x------->', x)
            # print('y------->', y)
            # print(testY0[d])
            # print(PatientName0_label[d])

            raise ValueError("x and y must have same first dimension, but "
                             "have shapes {} and {}".format(x.shape, y.shape))

    def lowess(x, y, reduction_factor=1. / 3.):

        xwidth = reduction_factor * (x.max() - x.min())
        observations = len(x)
        order = np.argsort(x)
        y_sm = np.zeros_like(y)
        y_stderr = np.zeros_like(y)
        tricube = lambda d: np.clip((1 - np.abs(d) ** 3) ** 3, 0, 1)
        for i in range(observations):
            dist = np.abs((x[order][i] - x[order])) / xwidth
            w = tricube(dist)
            A = np.stack([w, x[order] * w]).T
            b = w * y[order]
            ATA = A.T.dot(A)
            ATb = A.T.dot(b)
            sol = np.linalg.solve(ATA, ATb)
            yest = A[i].dot(sol)
            place = order[i]
            y_sm[place] = yest
            sigma2 = (np.sum((A.dot(sol) - y[order]) ** 2) / observations)
            y_stderr[place] = np.sqrt(sigma2 *
                                      A[i].dot(np.linalg.inv(ATA)
                                               ).dot(A[i]))
        return y_sm, y_stderr

    # <......................
    # y = np.array(X_arr)
    # x = np.array(Y_arr)
    # <......................
    print('\x1b[6;30;41m' + "                         " + '\x1b[0m')
    print('\x1b[6;30;41m' + "plotting negative sample:" + '\x1b[0m')
    print('\x1b[6;30;41m' + "                         " + '\x1b[0m')
    print(CBLUE + 'Texture feature:' + CEND, feature)

    y = np.array(X_arr0)
    x = np.array(Y_arr0)
    # print('inputx:', x)
    # print('inputy:', y)
    order = np.argsort(x)
    # run it
    y_sm, y_std = lowess(x, y)
    # plot it
    fig, ax = plt.subplots()
    ax.plot(x[order], y_sm[order], color='tomato', label='LOWESS')
    # 95% of the area under a normal curve lies within roughly 1.96 standard deviations
    # of the mean, and due to the central limit theorem,
    # this number is therefore used in the construction of approximate 95% confidence intervals
    ax.fill_between(x[order], y_sm[order] - 1.96 * y_std[order],
                    y_sm[order] + 1.96 * y_std[order], alpha=0.3, label='LOWESS uncertainty(1.96)')
    # plt.fill_between(x[order], y_sm[order] - y_std[order],
    #                  y_sm[order] + y_std[order], alpha=0.3, label='LOWESS uncertainty')
    ax.plot(x, y, 'k.', label='Observations')
    ax.axvline(x=0.0, linestyle='dashed', color='red')
    # for s in testX:
    #     l = np.array(testX[s])
    #     s = np.array(cleaned_listY[s])
    #     ax.plot(s, l, color='green', linewidth=0.75, linestyle='dashed', marker='o',
    #             markerfacecolor='blue', markersize=3.5)
    for s in testX0:
        l = np.array(testX0[s])
        s = np.array(cleaned_listY0[s])
        ax.plot(s, l, color='green', linewidth=0.75, linestyle='dashed', marker='o',
                markerfacecolor='blue', markersize=3.5)
    # 20171218 - 20162810
    #
    r = np.where(y == np.max(y))
    l = r[0][0]
    ax.annotate(
        np.max(y),
        xy=(x[l], y[l]), xytext=(-2, 2),
        textcoords='offset points', ha='right', va='bottom',
        # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', fc='yellow', connectionstyle='arc3,rad=0'))
    r = np.where(y == np.min(y))
    k = r[0][0]
    ax.annotate(
        np.min(y),
        xy=(x[k], y[k]), xytext=(-2, 2),
        textcoords='offset points', ha='right', va='bottom',
        # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', fc='yellow', connectionstyle='arc3,rad=0'))
    ax.legend(loc='upper right')
    ax.set_xlabel('Years To Diagnosis')
    ax.set_ylabel(feature)
    ax.set_title('Cancer Patients Diagnosis (negative)')
    ax.grid(True)
    # run it
    y_sm, y_std = lowess(x, y, reduction_factor=1. / 5.)
    # plot it
    plt.plot(x[order], y_sm[order], color='yellow', label='LOWESS(1/5)')
    plt.fill_between(x[order], y_sm[order] - y_std[order],
                     y_sm[order] + y_std[order], alpha=0.3, label='LOWESS uncertainty')
    plt.plot(x, y, 'k.', label='Observations')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(plot_save + feature + '_T2w_negative.png')
    plt.close()


    # <......................
    y = np.array(X_arr)
    x = np.array(Y_arr)
    # <......................
    print('\x1b[6;30;41m' + "                         " + '\x1b[0m')
    print('\x1b[6;30;41m' + "plotting positive sample:" + '\x1b[0m')
    print('\x1b[6;30;41m' + "                         " + '\x1b[0m')
    print(CBLUE + 'Texture feature:' + CEND, feature)

    # y = np.array(X_arr0)
    # x = np.array(Y_arr0)
    # print('inputx:', x)
    # print('inputy:', y)
    order = np.argsort(x)
    # run it
    y_sm, y_std = lowess(x, y)
    # plot it
    fig, ax = plt.subplots()
    ax.plot(x[order], y_sm[order], color='tomato', label='LOWESS')
    # 95% of the area under a normal curve lies within roughly 1.96 standard deviations
    # of the mean, and due to the central limit theorem,
    # this number is therefore used in the construction of approximate 95% confidence intervals
    ax.fill_between(x[order], y_sm[order] - 1.96 * y_std[order],
                    y_sm[order] + 1.96 * y_std[order], alpha=0.3, label='LOWESS uncertainty(1.96)')
    # plt.fill_between(x[order], y_sm[order] - y_std[order],
    #                  y_sm[order] + y_std[order], alpha=0.3, label='LOWESS uncertainty')
    ax.plot(x, y, 'k.', label='Observations')
    ax.axvline(x=0.0, linestyle='dashed', color='red')
    for s in testX:
        l = np.array(testX[s])
        s = np.array(cleaned_listY[s])
        ax.plot(s, l, color='green', linewidth=0.75, linestyle='dashed', marker='o',
                markerfacecolor='blue', markersize=3.5)
    # for s in testX0:
    #     l = np.array(testX0[s])
    #     s = np.array(cleaned_listY0[s])
    #     ax.plot(s, l, color='green', linewidth=0.75, linestyle='dashed', marker='o',
    #             markerfacecolor='blue', markersize=3.5)
    # 20171218 - 20162810
    #
    r = np.where(y == np.max(y))
    l = r[0][0]
    ax.annotate(
        np.max(y),
        xy=(x[l], y[l]), xytext=(-2, 2),
        textcoords='offset points', ha='right', va='bottom',
        # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', fc='yellow', connectionstyle='arc3,rad=0'))
    r = np.where(y == np.min(y))
    k = r[0][0]
    ax.annotate(
        np.min(y),
        xy=(x[k], y[k]), xytext=(-2, 2),
        textcoords='offset points', ha='right', va='bottom',
        # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', fc='yellow', connectionstyle='arc3,rad=0'))
    ax.legend(loc='upper right')
    ax.set_xlabel('Years To Diagnosis')
    ax.set_ylabel(feature)
    ax.set_title('Cancer Patients Diagnosis (positive)')
    ax.grid(True)
    # run it
    y_sm, y_std = lowess(x, y, reduction_factor=1. / 5.)
    # plot it
    plt.plot(x[order], y_sm[order], color='yellow', label='LOWESS(1/5)')
    plt.fill_between(x[order], y_sm[order] - y_std[order],
                     y_sm[order] + y_std[order], alpha=0.3, label='LOWESS uncertainty')
    plt.plot(x, y, 'k.', label='Observations')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(plot_save + feature + '_T2w_positive.png')
    plt.close()

dfx_T2w0= pd.read_csv('RobustRadiomics_T2w_Final_serial.csv', index_col=0, nrows=0).columns.tolist()

featureName = np.array(dfx_T2w0)
print('featureNames:', featureName)
for fn in featureName:
    # print('fn', fn)
    print(CWHITE + "-------------------------" + CEND)

    plot_all(feature=fn)


