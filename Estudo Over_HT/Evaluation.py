import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import make_scorer, recall_score, precision_score
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv('./Games.csv')

columns = ['Total Shots Home', 'Total Shots Away', 'Shots Insidebox Home', 'Shots Insidebox Away', 'Fouls Home', 'Fouls Away']

def func_aux(column):
    return pd.to_numeric(column.apply(lambda x: 0 if x == '-' else x))

df[columns] = df[columns].apply(func_aux)

columns = ['Odd Home', 'Odd Draw',	'Odd Away']

def func_aux(column):
    return pd.to_numeric(column.apply(lambda x: x.replace(',', '.')))

df[columns] = df[columns].apply(func_aux)

df['Total Actions'] = df['Total Shots Home'] + df['Total Shots Away'] + df['Shots Insidebox Home'] + df['Shots Insidebox Away'] + df['Fouls Home'] + df['Fouls Away']

df = df[df['Total Actions'] > 0]

df = df[df['League'] == 'Liga Profesional Argentina'] # apenas Liga Profesional Argentina

df.drop(['Game', 'League', 'Winner', 'Total Actions'], axis=1, inplace=True)

def get_outliers_indexes(column, IQR_CONST):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1 # diferente de .quantile(0.5)

    upper_bound = Q3 + IQR_CONST * IQR
    lower_bound = Q1 - IQR_CONST * IQR

    indexes = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index

    return indexes

scores = []

for IQR_CONST in [1, 1.25, 1.5, 1.75, 2]:
    print(IQR_CONST)
    df_new = df.copy()

    indexes = []

    for column in df_new:
        new_outliers = get_outliers_indexes(column, IQR_CONST)

        indexes.extend(new_outliers)

    indexes_without_duplicates = list(set(indexes))

    df_new.drop(indexes_without_duplicates, inplace=True)

    X = df_new.loc[:, df_new.columns != 'Over 0.5 HT']
    Y = df_new['Over 0.5 HT']

    X, Y = RandomUnderSampler(random_state=42).fit_resample(X, Y)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    for learning_rate in [x * 0.05 for x in range(1, 10, 2)]:
        for subsample in [x * 0.1 for x in range(1, 10, 2)]:
            for n_estimators in [1, 2, 4, 8, 16, 32, 64, 100, 200]:
                for max_depth in range(1, 5, 1):

                    model = GradientBoostingClassifier(
                        learning_rate=learning_rate,
                        subsample=subsample,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42,
                    )

                    kf = KFold(n_splits=5, random_state=42, shuffle=True)

                    # Accuracy/Precision

                    accuracy = cross_val_score(model, x_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1)

                    accuracy_1 = cross_val_score(model, x_train, y_train, cv=kf, scoring='precision', n_jobs=-1)

                    def accuracy_negative(y_true, y_pred):
                        return precision_score(y_true, y_pred, pos_label=0)

                    accuracy_0 = cross_val_score(model, x_train, y_train, cv=kf, scoring=make_scorer(accuracy_negative), n_jobs=-1)
                    
                    # Mean

                    accuracy_mean = accuracy.mean()

                    accuracy_1_mean = accuracy_1.mean()

                    accuracy_0_mean = accuracy_0.mean()

                    # Median

                    accuracy_median = np.median(accuracy)

                    accuracy_1_median = np.median(accuracy_1)

                    accuracy_0_median = np.median(accuracy_0)

                    # Recall Mean

                    recall_1_mean = cross_val_score(model, x_train, y_train, cv=kf, scoring='recall', n_jobs=-1).mean()

                    def recall_negative(y_true, y_pred):
                        return recall_score(y_true, y_pred, pos_label=0)
                    
                    recall_0_mean = cross_val_score(model, x_train, y_train, cv=kf, scoring=make_scorer(recall_negative), n_jobs=-1).mean()

                    scores.append([
                        IQR_CONST,
                        learning_rate,
                        subsample,
                        n_estimators,
                        max_depth,
                        accuracy_mean,
                        accuracy_1_mean,
                        accuracy_0_mean,
                        accuracy_median,
                        accuracy_1_median,
                        accuracy_0_median,
                        recall_1_mean,
                        recall_0_mean,
                    ])

columns = [
    'iqr',
    'learning_rate',
    'subsample',
    'n_estimators',
    'max_depth',
    'accuracy_mean',
    'accuracy_1_mean',
    'accuracy_0_mean',
    'accuracy_median',
    'accuracy_1_median',
    'accuracy_0_median',
    'recall_1_mean',
    'recall_0_mean'
]

df_scores = pd.DataFrame(scores, columns=columns)
df_scores.to_csv('70-30 (Liga Profesional Argentina).csv', index=False)