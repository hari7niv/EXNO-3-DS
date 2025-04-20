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

df=pd.read_csv("/content/Encoding Data.csv")

df

![image](https://github.com/user-attachments/assets/2e4f2994-10a5-46ee-926c-8ff85bc1d87a)

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

pm=['Hot','Warm','Cold']

e1=OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["ord_2"]])

![image](https://github.com/user-attachments/assets/761943cf-2c18-4ba4-8be2-a5baebb2aab6)

df['bo2']=e1.fit_transform(df[["ord_2"]])

df

![image](https://github.com/user-attachments/assets/bea4601f-a233-4685-b0cf-a92abb226882)

le=LabelEncoder()

dfc=df.copy()

dfc['ord_2']=le.fit_transform(dfc['ord_2'])

dfc

![image](https://github.com/user-attachments/assets/dda818bd-a698-47a9-a6df-3a0ab8e737d9)

from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(sparse_output=False)

df2=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))

df2=pd.concat([df2,enc],axis=1)

df2

![image](https://github.com/user-attachments/assets/276f8a49-0f1c-4804-8f0d-3ac0ef7c6feb)

pd.get_dummies(df2,columns=["nom_0"])

![image](https://github.com/user-attachments/assets/8d21f06b-3831-4876-a73c-e73637503879)

pip install --upgrade category_encoders

![image](https://github.com/user-attachments/assets/11db2d58-c43f-4d7f-93c1-d754b59d26c6)


from category_encoders import BinaryEncoder

be=BinaryEncoder()

df=pd.read_csv("/content/data.csv")

df

![image](https://github.com/user-attachments/assets/8078e91f-7a66-4e81-969a-ee52a1c49f3f)

be= BinaryEncoder()

nd=be.fit_transform(df['Ord_2'])

df=pd.concat([df,nd],axis=1)

dfb1=df.copy()

dfb1

![image](https://github.com/user-attachments/assets/72635e68-2afb-4c15-a066-c61a55882a08)


from category_encoders import TargetEncoder

te=TargetEncoder()

cc=df.copy()

new=te.fit_transform(X=cc["City"],y=cc["Target"])

cc=pd.concat([cc,new],axis=1)

cc

![image](https://github.com/user-attachments/assets/f12a1618-0925-405d-a98c-f58b62ba30a6)

import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("/content/Data_to_Transform.csv")

df

![image](https://github.com/user-attachments/assets/ea3f4aa4-abac-4d70-8a45-c42e3031cb3e)

df.skew()

![image](https://github.com/user-attachments/assets/5da1f669-86ab-4185-a17e-991ae4eac2cc)

np.log(df["Highly Positive Skew"])

![image](https://github.com/user-attachments/assets/1815a329-4eda-487a-b7b4-6b2a06546c33)

np.reciprocal(df["Moderate Positive Skew"])

![image](https://github.com/user-attachments/assets/168006d6-c505-4aed-a395-b7deb42924ed)

np.sqrt(df['Highly Positive Skew'])

![image](https://github.com/user-attachments/assets/b597ca2e-8dff-422e-998b-38edf4fa328b)

np.square(df["Highly Positive Skew"])

![image](https://github.com/user-attachments/assets/0367ee3a-af35-4f69-8d04-85d1f80e3458)

df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df['Highly Positive Skew'])

df

![image](https://github.com/user-attachments/assets/7f8d32e2-cb5b-49b7-b896-5c01e549c2c5)

df.skew()

![image](https://github.com/user-attachments/assets/0fc05814-6c28-49a7-a070-46abdf65ed08)

df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])

df.skew()

![image](https://github.com/user-attachments/assets/5f90a561-40af-4c75-b4c5-f284b7e2d29f)

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])

df

![image](https://github.com/user-attachments/assets/8291853e-5c25-4684-852d-34388b21914e)

import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()

![image](https://github.com/user-attachments/assets/e7f903f9-dbe3-4b46-94b9-7345794d04e4)

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')

plt.show()

![image](https://github.com/user-attachments/assets/e12550f8-0ff0-491f-b0b7-bd6badb4b8d7)

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()

![image](https://github.com/user-attachments/assets/170a8263-5cee-4620-82b2-4b1751c5500f)

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])

sm.qqplot(df["Highly Negative Skew"],line='45')

plt.show()

![image](https://github.com/user-attachments/assets/6b3b8ef2-bb29-44d9-b526-f410401b2dd2)


sm.qqplot(df["Highly Negative Skew_1"],line='45')

plt.show()

![image](https://github.com/user-attachments/assets/94e2a3be-e6b7-4d76-b527-b062f231c268)


dt=pd.read_csv("/content/titanic_dataset (2).csv")

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

dt["Age_1"]=qt.fit_transform(dt[["Age"]])

sm.qqplot(dt['Age'],line='45')

![image](https://github.com/user-attachments/assets/bb4d65f2-b2b2-439b-be65-cc8485157533)

sm.qqplot(dt['Age_1'],line='45')

plt.show()


![image](https://github.com/user-attachments/assets/a52c79c0-5a2f-4167-9497-828a7a602808)

# RESULT:

      Thus we performed Feature Encoding and Transformation process

       
