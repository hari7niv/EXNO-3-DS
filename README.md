## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
    import pandas as pd

    from scipy import stats

    import numpy as np

    from sklearn.preprocessing import StandardScaler

    df=pd.read_csv("/content/bmi.csv")

    df.head()

![image](https://github.com/user-attachments/assets/77dbc9bf-4472-4173-b787-654911d60bf8)


    df.dropna()

![image](https://github.com/user-attachments/assets/953ec500-73bf-4557-b179-f3a348c7cd2b)


    max_vals=np.max(np.abs(df[['Height','Weight']]))

    max_vals

![image](https://github.com/user-attachments/assets/89bbb24f-9409-4dc0-8c49-ed1ea5da786d)


    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()

    df=pd.read_csv("/content/bmi (1).csv")

    df[['Height', 'Weight']] = sc.fit_transform(df[['Height', 'Weight']])

    print(df.head(10))

![image](https://github.com/user-attachments/assets/ad41a79a-6cd4-4140-9d90-7360ae9f5c9c)


    from sklearn.preprocessing import MinMaxScaler

    scaler=MinMaxScaler()

    df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])

    df.head(10)

![image](https://github.com/user-attachments/assets/1ed09ea6-2b8d-4fd4-a320-c62b3bc594e0)


    from sklearn.preprocessing import Normalizer

    scaler=Normalizer()

    df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])

    df

![image](https://github.com/user-attachments/assets/1ab44a24-9cfe-4b91-bbfe-92b15e8ecefa)


    df3=pd.read_csv("/content/bmi.csv")

    from sklearn.preprocessing import MaxAbsScaler

    scaler=MaxAbsScaler()

    df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])

    df3

![image](https://github.com/user-attachments/assets/3e85af7b-a0cb-467c-91c7-3a1ac842e1e3)


    df4=pd.read_csv("/content/bmi.csv")

    from sklearn.preprocessing import RobustScaler

    scaler=RobustScaler()

    df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])

    df4.head()
  
![image](https://github.com/user-attachments/assets/cd2f677f-a7d8-47f7-a206-8c0699f73cec)



    from scipy.stats import chi2_contingency

    import seaborn as sns

    tips=sns.load_dataset('tips')

    tips.head()

![image](https://github.com/user-attachments/assets/937657c5-db1f-47cd-9c86-45d064198a68)


    contigency_table=pd.crosstab(tips['sex'],tips['time'])

    print(contigency_table)

![image](https://github.com/user-attachments/assets/d7ac1871-adb7-49d9-b2fa-c547d977be9c)


    chi2,p, _, _ = chi2_contingency(contigency_table)

    print(f"Chi-Square Statistic: {chi2}")

  print(f"P-value: {p}")

![image](https://github.com/user-attachments/assets/f11c97c3-1003-4dfe-a8ff-16c86dd780ca)


    import pandas as pd

    from sklearn.feature_selection import SelectKBest , mutual_info_classif,f_classif


    data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
    }

    df=pd.DataFrame(data)

    x=df[['Feature1','Feature3']]

    y=df['Target']

    selector=SelectKBest(score_func=mutual_info_classif,k=1)

    X_new=selector.fit_transform(x,y)

    selected_feature_indices=selector.get_support(indices=True)

    selected_features=X.columns[selected_feature_indices]

    print("Selected Features:")

    print(selected_features)

![image](https://github.com/user-attachments/assets/6cca2bd4-8ad2-4ffb-8919-f87e0f8f2a4d)


# RESULT:
       Thus we performed Feature Scaling and Feature Selection process
