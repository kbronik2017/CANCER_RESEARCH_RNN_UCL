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
from numpy import arange
from pandas import read_csv
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNetCV
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
import sklearn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import MultiTaskLassoCV
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
# load the dataset
file1 = 'RobustRadiomics_ADC_Scan1_serial.csv'
dataframe1 = read_csv(file1, header=None)
data1 = dataframe1.values
# X, y = data[:, :-1], data[:, -1]
X1 = data1[1:, 1:]
print('X1', X1)

file2 = 'RobustRadiomics_T2w_Scan1_serial.csv'
dataframe2 = read_csv(file2, header=None)
data2 = dataframe2.values
# X, y = data[:, :-1], data[:, -1]
X2 = data2[1:, 1:]
print('X2', X2)

finalX12 = np.concatenate((X1, X2), axis=1)



file1f = 'RobustRadiomics_ADC_Final_serial.csv'
dataframe1f = read_csv(file1f, header=None)
data1f = dataframe1f.values
# X, y = data[:, :-1], data[:, -1]
X1f = data1f[1:, 1:]
# print('X1f', X1f)

file2f = 'RobustRadiomics_T2w_Final_serial.csv'
dataframe2f = read_csv(file2f, header=None)
data2f = dataframe2f.values
# X, y = data[:, :-1], data[:, -1]
X2f = data2f[1:, 1:]
# print('X2f', X2f)
dfx_adc0_label = pd.read_csv('RobustRadiomics_ADC_Scan1_serial.csv')
PatientName0_label = dfx_adc0_label["PatientName"]
PatientName0_label = np.array(PatientName0_label)
print('PatientName', PatientName0_label)


finalX12f = np.concatenate((X1f, X2f), axis=1)
delta_radiomics= np.asfarray(finalX12f,float) - np.asfarray(finalX12,float)


print('delta_radiomics ', np.array(delta_radiomics).shape)
# print('delta_radiomics', delta_radiomics.__class__)
dfx = read_csv('Serial radiomics database send out_280921.csv')
print('\x1b[6;30;41m' + "                                " + '\x1b[0m')
print('\x1b[6;30;41m' + "Loading data into memory is done" + '\x1b[0m')
print('\x1b[6;30;41m' + "                                " + '\x1b[0m')
print(CBLUE +'DATA:'+ CEND, dfx)
ytrain = dfx["Histopathological progression (0=no, 1=yes)"]
ytrain = np.array(ytrain)
y = ytrain

# print('y', y)

cv = RepeatedKFold(n_splits=19, n_repeats=3, random_state=2652124)

alphas = np.logspace(-4, -0.5, 30)
model = ElasticNetCV(alphas=alphas,  fit_intercept=False, tol=1e-6,
                   max_iter=100000, positive=True)



# unbalanced_training_and_testing
Trainx = delta_radiomics[:66,:]
Trainy = y[:66]
Testx = delta_radiomics[66:,:]
Testy = y[66:]


model.fit(Trainx, Trainy)
# summarize chosen configuration
print('alpha: %f' % model.alpha_)


# make a prediction
def predict_print(model):
    all_predictions = []
    for i in range(len(Testx)):
            # t1 = Testx[i,:]
            t1= np.expand_dims(Testx[i], 0)
            # print('t1_shape', t1.shape)

            result = model.predict(t1)
            # print(' result',  result)
            all_predictions.append(result[0])

            print(
                "Predicted result: is: %.3f, Target result is: %.3f" %(result[0], Testy[i]))
    return   np.array(all_predictions)



print('Testx_shape', np.array(Testx).shape)
print('Testy_shape', np.array(Testy).shape)
predicted_results = predict_print(model)

# print('PatientName', PatientName0_label[38:])
# print('PatientName', len(PatientName0_label[38:]))
rescaled1 = np.array(PatientName0_label[66:])
# print('predicted_results', predicted_results)
print('PatientNames:', rescaled1)
import xlsxwriter
workbook = xlsxwriter.Workbook('delta radiomics_ElasticNet_unbalanced.xlsx')
worksheet = workbook.add_worksheet()

worksheet.write_column(0, 0, rescaled1)
worksheet.write_column(0, 1, predicted_results)
workbook.close()

ns_probs = [0 for _ in range(len(Testy))]
# print('ns_probs', ns_probs)
ns_auc = roc_auc_score(Testy, ns_probs)
lr_auc = roc_auc_score(Testy, predicted_results)
# summarize scores
print(CYELLOW +'No Skill: ROC AUC=%.3f'  % (ns_auc) + CEND)
print(CYELLOW +'Predicted: ROC AUC=%.3f'  % (lr_auc) + CEND)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(Testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(Testy, predicted_results)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Predicted')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
# plt.show()
plt.savefig('delta radiomics_ElasticNet_unbalanced.png')

