# readmission prediction using data (demographics and 19 features) obtained from Pabkin et al. https://github.com/apakbin94/ICU72hReadmissionMIMICIII
# Using other features to predict demographic


import pickle
import argparse
import math
import pandas as pd
import numpy as np
import progressbar
from datetime import datetime, timedelta
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import time
import warnings
warnings.filterwarnings('ignore')

import sys

preprocess_data = False
path_views = "local_mimic/views"
path_tables = "local_mimic/tables"

start_time = time.time()
if preprocess_data:
        cohort = pd.read_csv(path_views + "/df_MASTER_DATA_cleaned.csv")
        cohort.columns = cohort.columns.str.lower()
        print(cohort.columns.to_list())


        # change from resprate to  respiratory_rate_lastmsrmt, also for temp and heartrate
        cohort = cohort[['subject_id', 'hadm_id', 'age', 'ethnicity','admission_type', 'marital_status',
                         'insurance', 'religion', 'gender', 'language',

                         'potassium_score', 'arterial_bp_diastolic_mean', #'hepatic', 'cardiovascular', 'liver',
                         'albumin_score', 'bun_score_x', 'creatinine_score', 'sodium_score_x', 'bicarbonate_score',
                         'heart_rate_lastmsrmt', 'sysbp_score', 'temperature_lastmsrmt', 'respiratory_rate_lastmsrmt',
                         'spo2_mean', 'glucose_score', 'coagulation', 'sapsii', 'pao2fio2_score', 'sirs', 'sofa', 'apsiii',
                         'isreadmitted_24hrs', 'isreadmitted_48hrs', 'isreadmitted_72hrs', 'isreadmitted_7days',
                         'isreadmitted_30days', 'isreadmitted_bounceback', 'los'
                         ]]


        # final join
        df = cohort#pd.merge(icu_details, admissions, on=['subject_id', 'hadm_id'])
        print("df", df)
        df = df.loc[:,~df.columns.duplicated()] # remove duplicate age column

        #make age whole number i.e round it to remove extra
        df['age'] = round(df['age'])

        # round los to 2 decimal place
        df['los'] = round(df['los'], 2)

        # print(df['age'].value_counts().to_string())

        print(df.columns)
        # ['subject_id', 'hadm_id', 'age', 'ethnicity', 'los_hospital', 'los_icu',
        #        'gender', 'first_hosp_stay', 'first_icu_stay', 'los_target',
        #        'discharge_location', 'marital_status', 'insurance', 'religion']


        # label encode each attribute
        le = LabelEncoder()
        df["ethnicity"] = le.fit_transform(df["ethnicity"])
        # print("le.classes_", le.classes_) #prints classes
        # print(df["ethnicity"].value_counts())

        df["gender"] = le.fit_transform(df["gender"])
        # print("le.classes_", le.classes_) #prints classes
        # print(df["gender"].value_counts())

        # df["first_icu_stay"] = le.fit_transform(df["first_icu_stay"])
        df["admission_type"] = le.fit_transform(df["admission_type"])
        df["insurance"] = le.fit_transform(df["insurance"])

        # dealing with null values of marital status n religion b4 encoding
        df[pd.isnull(df["marital_status"])]  = 'NaN'
        # print(df["marital_status"].value_counts())
        df["marital_status"] = le.fit_transform(df["marital_status"])

        df[pd.isnull(df["religion"])]  = 'NaN'
        # print(df["religion"].value_counts())
        df["religion"] = le.fit_transform(df["religion"])

        df[pd.isnull(df["language"])]  = 'NaN'
        df["language"] = le.fit_transform(df["language"])


        # Removes the row where hadm_id is NaN
        df = df[df['hadm_id'] != 'NaN']


        # # We can select the 13600 i.e 1700 from each "class" Reduces prediction
        # df = df.sample(n=13600, random_state=1, weights=df["los_target"])

        # save df for sanitization:
        df.to_csv("local_mimic/views/processed_data_demo_19features_with_readmission.csv", index=False)

# sys.exit()

# do feature selection using recursive feature selection
from sklearn.feature_selection import RFE

df = pd.read_csv(path_views + "/processed_data_demo_19features_with_readmission.csv") #processed_df_MASTER_DATA_cleaned
# df_MASTER_DATA_cleaned_10percent
df.columns = df.columns.str.lower()
print(df.columns.to_list())


# #uncomment if running grom scratch for label encoder
# # label encode each attribute
# le = LabelEncoder()
# df["ethnicity"] = le.fit_transform(df["ethnicity"])
# # print("le.classes_", le.classes_) #prints classes
# # print(df["ethnicity"].value_counts())
#
# df["gender"] = le.fit_transform(df["gender"])
# # print("le.classes_", le.classes_) #prints classes
# # print(df["gender"].value_counts())
#
# df["insurance"] = le.fit_transform(df["insurance"])
#
# # dealing with null values of marital status n religion b4 encoding
# df[pd.isnull(df["marital_status"])]  = 'NaN'
# # print(df["marital_status"].value_counts())
# df["marital_status"] = le.fit_transform(df["marital_status"])
#
# df[pd.isnull(df["religion"])]  = 'NaN'
# # print(df["religion"].value_counts())
# df["religion"] = le.fit_transform(df["religion"])
#
# df.to_csv("local_mimic/views/processed_df_MASTER_DATA_cleaned.csv", index=False)




# Using other data to predict demographic
# uncomment if data is not only demographic
df = df.dropna(axis=0, subset=['hadm_id']) #drop null
print(len(df))
print(df["insurance"].head(10))

feature_list = [#'ethnicity', 'marital_status',
                #         'insurance', 'religion', 'gender',

'potassium_score', 'arterial_bp_diastolic_mean', #'hepatic', 'cardiovascular', 'liver',
                         'albumin_score', 'bun_score_x', 'creatinine_score', 'sodium_score_x', 'bicarbonate_score',
                         'heart_rate_lastmsrmt', 'sysbp_score', 'temperature_lastmsrmt', 'respiratory_rate_lastmsrmt',
                         'spo2_mean', 'glucose_score', 'coagulation', 'sapsii', 'pao2fio2_score', 'sirs', 'sofa', 'apsiii',
                        #  'potassium_score', 'hepatic', 'cardiovascular', 'liver',
                        # 'albumin_score', 'bun_score_x', 'creatinine_score', 'sodium_score_x', 'bicarbonate_score',
                        #  'heartrate', 'sysbp_score', 'temp', 'resprate', 'spo2_mean', 'glucose_score'
                ]


features = df[feature_list].to_numpy()



# ================================================

target = df[['isreadmitted_24hrs']].to_numpy()  # df[['isreadmitted_24hrs']].to_numpy()
target = np.reshape(target, -1)
print("Target", target)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# imputer for null values ("NaN") of religion and marital status with mean of the encoded label
imputer = SimpleImputer(strategy="most_frequent")#mean
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)



# Computing relative importance of each attribute / features using Extratree classifier
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier(random_state=0)
model.fit(X_train, y_train)
# print("model.feature_importances_", model.feature_importances_)
for (importance, feat) in zip(model.feature_importances_, feature_list):
    print(importance, "=======>", feat)

model = LogisticRegression(  class_weight="balanced", C=10, multi_class="ovr", solver="lbfgs", max_iter=1000)

from sklearn.feature_selection import RFE

rfe = RFE(model, 3)
rfe = rfe.fit(X_train, y_train)
print("rfe.support", rfe.support_)
print("rfe.ranking",rfe.ranking_)
print("Rank ==============> Feature")
for (rank,feat) in zip(rfe.ranking_, feature_list):
        print(rank,"=======>",feat)
model.fit(X_train, y_train)

# print("Trainnnnnn")
# preds = model.predict(X_train)
# print(y_train[:10], preds[:10])
# rmse = math.sqrt(metrics.mean_squared_error(y_train, preds))
# print("Mean age:", y_train.mean())
# print("Error:", rmse)
# print('R-Square',metrics.r2_score(y_train,preds))
# print("Adjusted r-square", 1-(1-metrics.r2_score(y_train, preds))*((len(X_train)-1)/(len(X_train)-len(X_train[0])-1)))
# if feat_list != "age":
#     accuracy = metrics.accuracy_score(y_train, preds)
#     print("accuracy", accuracy)


print("Testttttt")
preds = model.predict(X_test)
print(y_test[:10], preds[:10])
rmse = math.sqrt(metrics.mean_squared_error(y_test, preds))
print("Mean age:", y_test.mean())
print("Error:", rmse)
print('R-Square', metrics.r2_score(y_test, preds))
print("Adjusted r-square",
      1 - (1 - metrics.r2_score(y_test, preds)) * ((len(X_test) - 1) / (len(X_test) - len(X_test[0]) - 1)))

accuracy = metrics.accuracy_score(y_test, preds)
print("accuracy", accuracy)

# sys.exit()
# ===============================================================

f_list = ['age', 'ethnicity', 'marital_status','insurance', 'religion', 'gender']
# Insurance, marital status, ethnicity, gender, religion {logistic regression}

# Although no need for this as it does not affect result!
le = LabelEncoder()

# TODO Bin the age and predict based on the binning. Then u can use logistic regression!

# We group age into 5 classes:
# df['age'] = pd.cut(df.age, 5, ['very-young', 'young', 'normal', 'old', 'very-old'])
# cut_bins = [18, 25, 45, 65, 89, 315] #315 is 89+
# (18, 25), (25, 45), (45, 65), (65, 89), (89+, ).

df['age'] = pd.cut(df['age'], bins=np.linspace(0, 315, 5)) # equally spaced into 5 bins #accuracy= 0.74
# df['age'] = pd.cut(df['age'], bins=cut_bins) # spaced according to cut_bins #accuracy = 0.54

print(df["age"].head(10))
print("ddddddddd", df['age'].value_counts())

# transform age
df["age"] = le.fit_transform(df["age"])


# df["ethnicity"] = le.fit_transform(df["ethnicity"])
# # print("le.classes_", le.classes_) #prints classes
# # print(df["ethnicity"].value_counts())
#
# df["gender"] = le.fit_transform(df["gender"])
# # print("le.classes_", le.classes_) #prints classes
# # print(df["gender"].value_counts())
#
# df["insurance"] = le.fit_transform(df["insurance"])
#
# # print(df["marital_status"].value_counts())
# df["marital_status"] = le.fit_transform(df["marital_status"])
#
# df["religion"] = le.fit_transform(df["religion"])


# Loops for 5 times to get average and std
rand_state_list = [42, 10, 20, 0, 87] # some random number
for rand_state in rand_state_list:
    for feat_list in f_list:
        print("=========", feat_list)
        target = df[[feat_list]].to_numpy() #df[['isreadmitted_24hrs']].to_numpy()
        target = np.reshape(target, -1)
        print("Target", target)

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=rand_state)

        # reshape target
        y_train = np.reshape(y_train, -1)
        y_test = np.reshape(y_test, -1)
        # print("y_train", y_train, "y_test", y_test)

        # imputer for null values ("NaN") of religion and marital status with mean of the encoded label
        imputer = SimpleImputer(strategy="most_frequent")#mean
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # # Rmove scaling. It reduces prediction
        # # scaling
        # sc = StandardScaler()
        # X_train = sc.fit_transform(X_train)
        # X_test = sc.transform(X_test)

        # if feat_list == "agee":
        #     model = LinearRegression(normalize=True)

        model = LogisticRegression() #class_weight="balanced", C=10, multi_class="ovr", solver="lbfgs", max_iter=1000

        model.fit(X_train, y_train)

        # print("Trainnnnnn")
        # preds = model.predict(X_train)
        # print(y_train[:10], preds[:10])
        # rmse = math.sqrt(metrics.mean_squared_error(y_train, preds))
        # print("Mean age:", y_train.mean())
        # print("Error:", rmse)
        # print('R-Square',metrics.r2_score(y_train,preds))
        # print("Adjusted r-square", 1-(1-metrics.r2_score(y_train, preds))*((len(X_train)-1)/(len(X_train)-len(X_train[0])-1)))
        # if feat_list != "age":
        #     accuracy = metrics.accuracy_score(y_train, preds)
        #     print("accuracy", accuracy)


        print("Testttttt")
        preds = model.predict(X_test)
        print(y_test[:10], preds[:10])
        rmse = math.sqrt(metrics.mean_squared_error(y_test, preds))
        print("Mean age:", y_test.mean())
        print("Error:", rmse)
        print('R-Square',metrics.r2_score(y_test,preds))
        print("Adjusted r-square", 1-(1-metrics.r2_score(y_test, preds))*((len(X_test)-1)/(len(X_test)-len(X_test[0])-1)))
        # if feat_list != "agee":
        accuracy = metrics.accuracy_score(y_test, preds)
        print("accuracy", accuracy)

sys.exit()








feature_list = ['age', 'ethnicity', 'marital_status',
                         'insurance', 'religion', 'gender',

                         #'potassium_score', 'hepatic', 'cardiovascular', 'liver',
                        #'albumin_score', 'bun_score_x', 'creatinine_score', 'sodium_score_x', 'bicarbonate_score',
                         #'heartrate', 'sysbp_score', 'temp', 'resprate', 'spo2_mean', 'glucose_score'
                ]

features = df[feature_list].to_numpy()
for targ in ["24hrs", "48hrs", "72hrs", "7days", "30days", "bounceback"]:
        targ_data = "isreadmitted_"+targ
        print("=====TargetData:", targ_data, "==========")
        target = df[[targ_data]].to_numpy() #df[['isreadmitted_24hrs']].to_numpy()
        # print("target.dtype", target.dtype)
        target = np.reshape(target, -1)
        # print("target.dtype", target.dtype)

        # print("np.any(np.isnan(features))",np.any(np.isnan(features)))
        # print("np.any(np.isnan(target))", np.any(np.isnan(target)))
        # print("features", features)
        # print("targetN", target)
        # print(feature_list)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

        # reshape target
        y_train = np.reshape(y_train, -1)
        y_test = np.reshape(y_test, -1)
        # print("y_train", y_train, "y_test", y_test)

        # imputer for null values ("NaN") of religion and marital status with mean of the encoded label
        imputer = SimpleImputer(strategy="mean")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # # Rmove scaling. It reduces prediction
        # # scaling
        # sc = StandardScaler()
        # X_train = sc.fit_transform(X_train)
        # X_test = sc.transform(X_test)

        # print("len(X_train)",len(X_train), "len(X_test)",len(X_test))

        model = LogisticRegression(class_weight="balanced", C=10, multi_class="ovr", solver="lbfgs", max_iter=1000)
        # model = LogisticRegression(multi_class="ovr", solver="lbfgs",max_iter=1000)
        # setting  ‘ovr’, then a binary problem is fit for each label improved from 0.76 to 0.8
        # If removing los_hospital, then remove class_weight="balanced". i.e use default! for LR. This increases accuracy from 0.28 to 0.34

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        # print("sjdjds", model.predict_proba(X_test)[:10, 1])
        f1_score = metrics.f1_score(y_test, preds, average="weighted")
        print("F1 score", f1_score)
        accuracy = metrics.accuracy_score(y_test, preds)
        print("accuracy",accuracy)
        auroc = metrics.roc_auc_score(y_test, preds)
        print("AUROC score", auroc)

end_time = time.time()
total_time = end_time - start_time
print("total time", total_time)

rfe = RFE(model, 3)
rfe = rfe.fit(X_train, y_train)
print("rfe.support", rfe.support_)
print("rfe.ranking",rfe.ranking_)
print("Rank ==============> Feature")
for (rank,feat) in zip(rfe.ranking_, feature_list):
        print(rank,"=======>",feat)



# Computing relative importance of each attribute / features using Extratree classifier
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier(random_state=0)
model.fit(X_train, y_train)
# print("model.feature_importances_", model.feature_importances_)
for (importance,feat) in zip(model.feature_importances_, feature_list):
        print(importance,"=======>",feat)







# # 100b (making all data have the same value i.e have the same value for each of the QIDs. i.e 100% generalization)
# # k100c and k10c are the ones that I suppressed 70%
# # k10d is with los_hospital generalized
#
# for data in ["","k10","k10c", "k10d", "k20", "k50", "k100", "k100b", "k100c", "k100d"]:
#         # "" ==> original
#         print("Data:", data)
#         df = pd.read_csv(path_views + "/processed_data"+data+".csv")
#
#         if data == "k10c" or data =="k100c":
#                 # completely remove data with *
#                 df = df[~df.age.str.contains("\*", na=False)]
#
#         # Adding los_hospital increases the model prediction. This is cos it somewhat correlates with the target preidction.
#         # If it is taken out, then we have 0.28 accuracy instead of 0.8. los_icu is also important from 0.24 to 0.31.
#         # Removing first_hosp_stay increases accuracy from 0.24 to o.28
#         #Religion adds and insurance adds nothing to the prediction
#         # age and marital_status is also less significant like 0.001
#         # adds nothing ethnicity
#
#
#
#         # Train model
#
#         # features = df[['age', 'ethnicity', 'los_icu', 'los_hospital',
#         #                'gender', 'first_hosp_stay', 'first_icu_stay', 'discharge_location',
#         #                'marital_status', 'insurance', 'religion']].to_numpy()
#
#         features = df[['age', 'ethnicity', 'los_icu', 'los_hospital',
#                 'gender', 'first_hosp_stay', 'first_icu_stay', 'discharge_location',
#                 'marital_status', 'insurance', 'religion'
#                         ]].to_numpy()
#         target = df[['los_target']].to_numpy()
#         # print("target.dtype", target.dtype)
#         target = np.reshape(target, -1)
#         # # print("target.dtype", set(target))
#         # print("features", features)
#         # print("targetN", target)
#
#         # Main problem is from the target. There is a null value in the target
#
#         X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)
#
#         # reshape target
#         y_train = np.reshape(y_train, -1)
#         y_test = np.reshape(y_test, -1)
#         # print("y_train", y_train, "y_test", y_test)
#
#         # imputer for null values ("NaN") of religion and marital status with mean of the encoded label
#         imputer = SimpleImputer(strategy="mean")
#         X_train = imputer.fit_transform(X_train)
#         X_test = imputer.transform(X_test)
#
#         # # Rmove scaling. It reduces prediction
#         # # scaling
#         # sc = StandardScaler()
#         # X_train = sc.fit_transform(X_train)
#         # X_test = sc.transform(X_test)
#
#         # print("len(X_train)",len(X_train), "len(X_test)",len(X_test))
#
#         model = LogisticRegression(class_weight="balanced", C=10, multi_class="ovr", solver="lbfgs", max_iter=1000)
#         # model = LogisticRegression(multi_class="ovr", solver="lbfgs",max_iter=1000)
#         # setting  ‘ovr’, then a binary problem is fit for each label improved from 0.76 to 0.8
#         # If removing los_hospital, then remove class_weight="balanced". i.e use default! for LR. This increases accuracy from 0.28 to 0.34
#
#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)
#         accuracy = metrics.accuracy_score(y_test, preds)
#         print("accuracy",accuracy)
#         f1_score = metrics.f1_score(y_test, preds, average="weighted")
#         print("F1 score", f1_score)
#
#
#         # # model = MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes=(3,),
#         # #                     learning_rate_init=5e-5, max_iter=5000, random_state=42)
#         #
#         # model = MLPClassifier(hidden_layer_sizes=(5000,), max_iter=10000,activation = 'relu',solver='adam',random_state=1)
#         #
#         # model.fit(X_train, y_train)
#         # preds = model.predict(X_test)
#         # accuracy = metrics.accuracy_score(y_test, preds)
#         # print("accuracy",accuracy)
#         # f1_score = metrics.f1_score(y_test, preds, average="weighted")
#         # print("F1 score", f1_score)
#
#
#
#
# # ##### Labels #####
# # def make_labels():
# #     icu_details = pd.read_csv(path_views + '/icustay_detail.csv')
# #     # apply exclusion criterias
# #     icu_details = icu_details[(icu_details.age >= 18) & (icu_details.los_hospital >= 1) & (icu_details.los_icu >= 1)]
# #     subj = list(set(icu_details.subject_id.tolist()))
# #     # make pivot tables for ICD-9
# #     print("=" * 80)
# #     print("Making pivot table for ICD-9 codes.".center(80))
# #     print("=" * 80)
# #     dx_dct, dx_freq = pivot_icd(subj)
# #     top25 = dx_freq[0:19] + dx_freq[20:26]
# #     top25 = [i[0] for i in top25]
# #     icd2idx = dict([(v, k) for k, v in enumerate(top25)])
# #     # make labels
# #     dct = {}
# #     bins = np.array([1, 2, 3, 5, 8, 14, 21, 30, 5000])
# #     print('Done!')
# #     print("=" * 80)
# #     print("Generating Labels...".center(80))
# #     print("=" * 80)
# #     for sample in progressbar.progressbar(range(len(subj))):
# #         s = subj[sample]
# #         lst = icu_details[icu_details.subject_id == s].hadm_id.tolist()
# #
# #         times = [(pd.to_datetime(icu_details[icu_details.hadm_id == i].admittime.values[0]),
# #                   pd.to_datetime(icu_details[icu_details.hadm_id == i].dischtime.values[0]), i) for i in lst]
# #         times = list(set(times))
# #         times = sorted(times, key=lambda x: x[0])
# #
# #         readmit = 0
# #         for t1, t2 in pairwise(iterable=times):
# #             difference = (t2[0] - t1[1]).days
# #             if difference <= 30:
# #                 hadm = t1[-1]
# #                 readmit = 1
# #             if difference < 0:
# #                 print(difference, s)
# #         if readmit == 0:
# #             morts = [(icu_details[icu_details.hadm_id == h[-1]].hospital_expire_flag.values[0], h[-1]) for h in times]
# #             hadm = [m[-1] for m in morts if m[0] == 1]
# #             if len(hadm) > 1:
# #                 print(morts)  # error, one can only experience mortlaity once
# #             elif len(hadm) == 1:
# #                 hadm = hadm[0]  # pick the mortality stay if no readmission
# #             else:
# #                 lengths = [(t[1] - t[0], t[-1]) for t in times]
# #                 hadm = sorted(lengths, key=lambda x: x[0])[-1][-1]  # pick the longest stay if no readmit and no deaths.
# #
# #         # digitize los
# #         los_bin = np.digitize(icu_details[(icu_details.hadm_id == hadm)].los_hospital.values[0], bins)
# #         # diagnostic labels
# #         dx_labels = [note for note in dx_dct[s][hadm] if note in top25]
# #         ohv = np.sum(one_hot([icd2idx[note] for note in dx_labels], 25), axis=0)
# #         dct[s] = {'hadm_id': hadm, 'readmit': readmit,
# #                   'los_hospital': icu_details[(icu_details.hadm_id == hadm)].los_hospital.values[0],
# #                   'los_bin': los_bin,
# #                   'mort': icu_details[icu_details.hadm_id == hadm].hospital_expire_flag.values[0],
# #                   'dx_lst': dx_dct[s][hadm],
# #                   'dx': ohv}
# #     return dct, dx_freq, dx_dct
# #
# #
# #
# #
# # def get_demographics(patients):
# #     '''patients: {subject_id: hadm_id}
# #     post: creates demographics dictionary by subject_id, and index dictionary'''
# #     from sklearn.preprocessing import LabelEncoder
# #     subj = list(set(patients.keys()))
# #     hadm = list(set(patients.values()))
# #     cohort = pd.read_csv(path_views + '/icustay_detail.csv')
# #     ## Exclusion criteria ##
# #     cohort = cohort[cohort.subject_id.isin(patients.keys()) & (cohort.hadm_id.isin(patients.values()))]
# #     admissions = pd.read_csv(path_tables + '/admissions.csv')
# #     cohort = cohort[['subject_id', 'hadm_id', 'age', 'ethnicity']]
# #     admissions = admissions[['subject_id', 'hadm_id', 'discharge_location', 'marital_status', 'insurance']]
# #     df = pd.merge(cohort, admissions, on=['subject_id', 'hadm_id'])
# #     df = df.drop_duplicates()
# #     df = df[(df.subject_id.isin(subj) & (df.hadm_id.isin(hadm)))]
# #     # discretize and to dict
# #     # df = df.set_index('subject_id')
# #     df = df.drop(columns=['hadm_id'])
# #     df['age'] = pd.qcut(df.age, 5, ['very-young', 'young', 'normal', 'old', 'very-old'])
# #     df['marital_status'] = df['marital_status'].fillna(value='UNKNOWN MARITAL')
# #     # make index unique
# #     df = df.groupby(['subject_id']).first().reset_index()
# #     df = df.set_index('subject_id')
# #     dct = df.to_dict('index')
# #     dct = dict([(k, list(set(v.values()))) for k, v in dct.items()])
# #     # label encoding
# #     categories = list(set(flatten([list(df[c].unique()) for c in list(df.columns)])))
# #     encoder = LabelEncoder()
# #     encoder.fit(categories)
# #     # label encode the dictionary
# #     dct = dict([(k, encoder.transform(v)) for k, v in dct.items()])
# #     category_dict = dict([(encoder.transform([c])[0], c) for c in categories])
# #     return dct, category_dict