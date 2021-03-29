# Using only demographics information for prediction


import pickle
import argparse
import pandas as pd
import numpy as np
import progressbar
from datetime import datetime, timedelta
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

preprocess_data = False
path_views = "local_mimic/views"
path_tables = "local_mimic/tables"

if preprocess_data:
        cohort = pd.read_csv(path_views + "/icustay_detail.csv")
        admissions = pd.read_csv(path_tables + "/admissions.csv")

        # used ethnicity_grouped instead of ethnicity so as to remove some other tribes that are not entirely necessary
        cohort = cohort[['subject_id', 'hadm_id', 'age', 'ethnicity_grouped', "los_hospital", "los_icu", "gender", "first_hosp_stay", "first_icu_stay"]]
        admissions = admissions[['subject_id', 'hadm_id', 'discharge_location', 'marital_status', 'insurance', 'religion']]

        # Merge by hadm_id

        # Target = los. Round it up!

        # make labels
        dct = {}
        bins = np.array([1, 2, 3, 5, 8, 14, 21, 30, 5000])

        icu_detailss = cohort[(cohort.age >= 18) & (cohort.los_hospital >= 1) & (cohort.los_icu >= 1)]
        icu_details = icu_detailss.copy()
        print("icu_details",icu_details)

        print('baba', icu_details["los_hospital"])


        # icu_details["los_hospital"] = icu_details["los_hospital"].fillna(icu_details["los_hospital"].mean())
        # sub2['income'].fillna((sub2['income'].mean()), inplace=True)

        # replace null values in los_hospital with the mean
        # print("icu_details.los_hospital.nunique()",icu_details.los_hospital.nunique()) #22125
        # print("icu_details.los_hospital.nunique() drop nan",icu_details.los_hospital.nunique(dropna=False)) #22125
        # digitize los
        los_bin = np.digitize(icu_details.los_hospital, bins)

        icu_details["los_target"] = los_bin

        print("icu_details.los_target.nunique()",icu_details.los_target.nunique()) #8
        print("icu_details.los_target.nunique() drop nan",icu_details.los_target.nunique(dropna=False)) #8

        print(len(los_bin))
        print("np.unique(los_bin)", np.unique(los_bin))

        print("icu_details2",icu_details)

        # x = np.array([0.2, 6.4, 3.0, 1.6, 9.0])
        # bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        # inds = np.digitize(x, bins) # the way it works is that the bin should start from the least, then the ind will begin from1 else 0
        # print("inds", inds)
        #
        #
        # for n in range(x.size):
        #   print(bins[inds[n]-1], "<=", x[n], "<", bins[inds[n]])

        # final join
        df = pd.merge(icu_details, admissions, on=['subject_id', 'hadm_id'])
        print("df", df)

        # # No duplicates though
        # df = df.drop_duplicates()
        # print("df2", df.to_string())

        # count how many records have
        print("df['los_target']",df['los_target'].value_counts())

        # 5    10849 ==> 8-14days
        # 4    10559 ==> 5-8days
        # 3     7364 ==> 3-5days
        # 6     5264 ==> 14-21days
        # 7     3151 ==> 21-30days
        # 8     3113 ==> 30+days
        # 2     2696 ==> 2-3days
        # 1     1713 ==> 1-2days

        #make age whole number i.e round it to remove extra
        df['age'] = round(df['age'])

        # print(df['age'].value_counts().to_string())

        print(df.columns)
        # ['subject_id', 'hadm_id', 'age', 'ethnicity_grouped', 'los_hospital', 'los_icu',
        #        'gender', 'first_hosp_stay', 'first_icu_stay', 'los_target',
        #        'discharge_location', 'marital_status', 'insurance', 'religion']


        # label encode each attribute
        le = LabelEncoder()
        df["ethnicity_grouped"] = le.fit_transform(df["ethnicity_grouped"])
        # print("le.classes_", le.classes_) #prints classes
        # print(df["ethnicity_grouped"].value_counts())

        df["gender"] = le.fit_transform(df["gender"])
        # print("le.classes_", le.classes_) #prints classes
        # print(df["gender"].value_counts())

        df["first_hosp_stay"] = le.fit_transform(df["first_hosp_stay"])
        df["first_icu_stay"] = le.fit_transform(df["first_icu_stay"])
        df["discharge_location"] = le.fit_transform(df["discharge_location"])
        df["insurance"] = le.fit_transform(df["insurance"])

        # dealing with null values of marital status n religion b4 encoding
        df[pd.isnull(df["marital_status"])]  = 'NaN'
        # print(df["marital_status"].value_counts())
        df["marital_status"] = le.fit_transform(df["marital_status"])

        df[pd.isnull(df["religion"])]  = 'NaN'
        # print(df["religion"].value_counts())
        df["religion"] = le.fit_transform(df["religion"])


        # Removes the row where los_target is NaN and converts to numeric
        df = df[df['los_target'] != 'NaN']
        df["los_target"] = pd.to_numeric(df["los_target"])

        # # We can select the 13600 i.e 1700 from each "class" Reduces prediction
        # df = df.sample(n=13600, random_state=1, weights=df["los_target"])

        # save df for sanitization:
        df.to_csv("local_mimic/views/processed_data.csv", index=False)


# do feature selection using recursive feature selection
from sklearn.feature_selection import RFE

df = pd.read_csv(path_views + "/processed_data.csv")
print(len(df))
# Train model

features = df[['age', 'ethnicity_grouped', 'los_icu', 'los_hospital',
        'gender', 'first_hosp_stay', 'first_icu_stay', 'discharge_location',
        'marital_status', 'insurance', 'religion'
                ]].to_numpy()
target = df[['los_target']].to_numpy()
# print("target.dtype", target.dtype)
target = np.reshape(target, -1)
# # print("target.dtype", set(target))
# print("features", features)
# print("targetN", target)


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

# model = LogisticRegression(class_weight="balanced", C=10, multi_class="ovr", solver="lbfgs", max_iter=1000)
model = LogisticRegression(multi_class="ovr", solver="lbfgs",max_iter=1000)

rfe = RFE(model, 3)
rfe = rfe.fit(X_train, y_train)
print("rfe.support", rfe.support_)
print("rfe.ranking",rfe.ranking_)
print("Rank ==============> Feature")
for (rank,feat) in zip(rfe.ranking_, ['age', 'ethnicity_grouped', 'los_icu', 'los_hospital', 'gender', 'first_hosp_stay',
      'first_icu_stay', 'discharge_location', 'marital_status', 'insurance', 'religion']):
        print(rank,"=======>",feat)



# Computing relative importance of each attribute / features using Extratree classifier
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier(random_state=0)
model.fit(X_train, y_train)
# print("model.feature_importances_", model.feature_importances_)
for (importance,feat) in zip(model.feature_importances_, ['age', 'ethnicity_grouped', 'los_icu', 'los_hospital', 'gender', 'first_hosp_stay',
      'first_icu_stay', 'discharge_location', 'marital_status', 'insurance', 'religion']):
        print(importance,"=======>",feat)







# 100b (making all data have the same value i.e have the same value for each of the QIDs. i.e 100% generalization)
# k100c and k10c are the ones that I suppressed 70%
# k10d is with los_hospital generalized

for data in ["","k10","k10c", "k10d", "k20", "k50", "k100", "k100b", "k100c", "k100d"]:
        # "" ==> original
        print("Data:", data)
        df = pd.read_csv(path_views + "/processed_data"+data+".csv")

        if data == "k10c" or data =="k100c":
                # completely remove data with *
                df = df[~df.age.str.contains("\*", na=False)]

        # Adding los_hospital increases the model prediction. This is cos it somewhat correlates with the target preidction.
        # If it is taken out, then we have 0.28 accuracy instead of 0.8. los_icu is also important from 0.24 to 0.31.
        # Removing first_hosp_stay increases accuracy from 0.24 to o.28
        #Religion adds and insurance adds nothing to the prediction
        # age and marital_status is also less significant like 0.001
        # adds nothing ethnicity_grouped



        # Train model

        # features = df[['age', 'ethnicity_grouped', 'los_icu', 'los_hospital',
        #                'gender', 'first_hosp_stay', 'first_icu_stay', 'discharge_location',
        #                'marital_status', 'insurance', 'religion']].to_numpy()

        features = df[['age', 'ethnicity_grouped', 'los_icu', 'los_hospital',
                'gender', 'first_hosp_stay', 'first_icu_stay', 'discharge_location',
                'marital_status', 'insurance', 'religion'
                        ]].to_numpy()
        target = df[['los_target']].to_numpy()
        # print("target.dtype", target.dtype)
        target = np.reshape(target, -1)
        # # print("target.dtype", set(target))
        # print("features", features)
        # print("targetN", target)

        # Main problem is from the target. There is a null value in the target

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
        accuracy = metrics.accuracy_score(y_test, preds)
        print("accuracy",accuracy)
        f1_score = metrics.f1_score(y_test, preds, average="weighted")
        print("F1 score", f1_score)


        # # model = MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes=(3,),
        # #                     learning_rate_init=5e-5, max_iter=5000, random_state=42)
        #
        # model = MLPClassifier(hidden_layer_sizes=(5000,), max_iter=10000,activation = 'relu',solver='adam',random_state=1)
        #
        # model.fit(X_train, y_train)
        # preds = model.predict(X_test)
        # accuracy = metrics.accuracy_score(y_test, preds)
        # print("accuracy",accuracy)
        # f1_score = metrics.f1_score(y_test, preds, average="weighted")
        # print("F1 score", f1_score)




# ##### Labels #####
# def make_labels():
#     icu_details = pd.read_csv(path_views + '/icustay_detail.csv')
#     # apply exclusion criterias
#     icu_details = icu_details[(icu_details.age >= 18) & (icu_details.los_hospital >= 1) & (icu_details.los_icu >= 1)]
#     subj = list(set(icu_details.subject_id.tolist()))
#     # make pivot tables for ICD-9
#     print("=" * 80)
#     print("Making pivot table for ICD-9 codes.".center(80))
#     print("=" * 80)
#     dx_dct, dx_freq = pivot_icd(subj)
#     top25 = dx_freq[0:19] + dx_freq[20:26]
#     top25 = [i[0] for i in top25]
#     icd2idx = dict([(v, k) for k, v in enumerate(top25)])
#     # make labels
#     dct = {}
#     bins = np.array([1, 2, 3, 5, 8, 14, 21, 30, 5000])
#     print('Done!')
#     print("=" * 80)
#     print("Generating Labels...".center(80))
#     print("=" * 80)
#     for sample in progressbar.progressbar(range(len(subj))):
#         s = subj[sample]
#         lst = icu_details[icu_details.subject_id == s].hadm_id.tolist()
#
#         times = [(pd.to_datetime(icu_details[icu_details.hadm_id == i].admittime.values[0]),
#                   pd.to_datetime(icu_details[icu_details.hadm_id == i].dischtime.values[0]), i) for i in lst]
#         times = list(set(times))
#         times = sorted(times, key=lambda x: x[0])
#
#         readmit = 0
#         for t1, t2 in pairwise(iterable=times):
#             difference = (t2[0] - t1[1]).days
#             if difference <= 30:
#                 hadm = t1[-1]
#                 readmit = 1
#             if difference < 0:
#                 print(difference, s)
#         if readmit == 0:
#             morts = [(icu_details[icu_details.hadm_id == h[-1]].hospital_expire_flag.values[0], h[-1]) for h in times]
#             hadm = [m[-1] for m in morts if m[0] == 1]
#             if len(hadm) > 1:
#                 print(morts)  # error, one can only experience mortlaity once
#             elif len(hadm) == 1:
#                 hadm = hadm[0]  # pick the mortality stay if no readmission
#             else:
#                 lengths = [(t[1] - t[0], t[-1]) for t in times]
#                 hadm = sorted(lengths, key=lambda x: x[0])[-1][-1]  # pick the longest stay if no readmit and no deaths.
#
#         # digitize los
#         los_bin = np.digitize(icu_details[(icu_details.hadm_id == hadm)].los_hospital.values[0], bins)
#         # diagnostic labels
#         dx_labels = [note for note in dx_dct[s][hadm] if note in top25]
#         ohv = np.sum(one_hot([icd2idx[note] for note in dx_labels], 25), axis=0)
#         dct[s] = {'hadm_id': hadm, 'readmit': readmit,
#                   'los_hospital': icu_details[(icu_details.hadm_id == hadm)].los_hospital.values[0],
#                   'los_bin': los_bin,
#                   'mort': icu_details[icu_details.hadm_id == hadm].hospital_expire_flag.values[0],
#                   'dx_lst': dx_dct[s][hadm],
#                   'dx': ohv}
#     return dct, dx_freq, dx_dct
#
#
#
#
# def get_demographics(patients):
#     '''patients: {subject_id: hadm_id}
#     post: creates demographics dictionary by subject_id, and index dictionary'''
#     from sklearn.preprocessing import LabelEncoder
#     subj = list(set(patients.keys()))
#     hadm = list(set(patients.values()))
#     cohort = pd.read_csv(path_views + '/icustay_detail.csv')
#     ## Exclusion criteria ##
#     cohort = cohort[cohort.subject_id.isin(patients.keys()) & (cohort.hadm_id.isin(patients.values()))]
#     admissions = pd.read_csv(path_tables + '/admissions.csv')
#     cohort = cohort[['subject_id', 'hadm_id', 'age', 'ethnicity_grouped']]
#     admissions = admissions[['subject_id', 'hadm_id', 'discharge_location', 'marital_status', 'insurance']]
#     df = pd.merge(cohort, admissions, on=['subject_id', 'hadm_id'])
#     df = df.drop_duplicates()
#     df = df[(df.subject_id.isin(subj) & (df.hadm_id.isin(hadm)))]
#     # discretize and to dict
#     # df = df.set_index('subject_id')
#     df = df.drop(columns=['hadm_id'])
#     df['age'] = pd.qcut(df.age, 5, ['very-young', 'young', 'normal', 'old', 'very-old'])
#     df['marital_status'] = df['marital_status'].fillna(value='UNKNOWN MARITAL')
#     # make index unique
#     df = df.groupby(['subject_id']).first().reset_index()
#     df = df.set_index('subject_id')
#     dct = df.to_dict('index')
#     dct = dict([(k, list(set(v.values()))) for k, v in dct.items()])
#     # label encoding
#     categories = list(set(flatten([list(df[c].unique()) for c in list(df.columns)])))
#     encoder = LabelEncoder()
#     encoder.fit(categories)
#     # label encode the dictionary
#     dct = dict([(k, encoder.transform(v)) for k, v in dct.items()])
#     category_dict = dict([(encoder.transform([c])[0], c) for c in categories])
#     return dct, category_dict