import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib


df=pd.read_csv("../../data/heart_2020_cleaned.csv")


t=df['BMI']
cond = [(t.between(0,18.5)),(t.between(18.5,24.9)),(t.between(24.9,29.9)),(t.between(29.9,34.9)), (t.between(34.9,100))]

labels = ['Uwt','N','Owt','O','EO']    # Uwt-underweight, N-normal, Owt-overweight, O-Obese, EO-extremely obese 

df['BMI_cat'] = np.select(cond,labels)


st=df['SleepTime']
cond = [(st.between(0,6)),(st.between(6,9)),(st.between(9,24))]

labels = ['Low','Normal','High']
df['SleepTime_cat'] = np.select(cond,labels)


AgeCategory_mean = {'18-24':21,'25-29':27,'30-34':32,'35-39':37,'40-44':42,'45-49':47,'50-54':52,'55-59':57, 
                    '60-64':62,'65-69':67,'70-74':72,'75-79':77,'80 or older':80}

df['Mean_Age'] = df['AgeCategory'].apply(lambda x: AgeCategory_mean[x])





df.drop(columns=['AlcoholDrinking','MentalHealth', 'BMI','SleepTime','AgeCategory'], inplace=True)



##Label Encoding
le = LabelEncoder()
cols=['HeartDisease', 'Smoking', 'Stroke', 'DiffWalking', 'Sex', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']

for i in cols:
    df[i] = le.fit_transform(df[i])

le_1=le.fit(df['GenHealth'])
le_1.classes_ = np.array(['Poor', 'Fair','Good','Very good','Excellent'])   ## to assign 0 to Poor and 4 to Excellent
df['GenHealth'] = le_1.transform(df['GenHealth'])

## One hot encoding
df=pd.concat([df,pd.get_dummies(df['Race'],prefix='Race',drop_first=True)], axis=1)
df=pd.concat([df,pd.get_dummies(df['Diabetic'],prefix='Diabetic',drop_first=True)], axis=1)
df=pd.concat([df,pd.get_dummies(df['BMI_cat'],prefix='BMI',drop_first=True)], axis=1)
df=pd.concat([df,pd.get_dummies(df['SleepTime_cat'],prefix='SleepTime',drop_first=True)], axis=1)

df.drop(columns=['Race','Diabetic','BMI_cat','SleepTime_cat'],axis=1,inplace=True)

X=df.drop(columns=['HeartDisease'],axis=1)
y=df['HeartDisease']

col=X.columns

sc = StandardScaler()

scaler=sc.fit(X)
X_scaled = scaler.transform(X)

X_train = pd.DataFrame(X_scaled, columns = col)

joblib.dump(scaler,'../model/scaler.pkl')  ## saving the scaler

model_rf = RandomForestClassifier(random_state=42)
model_rf.set_params(n_estimators=500, class_weight="balanced",max_depth=30,min_samples_split= 30,min_samples_leaf=24)
model_rf.fit(X_train,y)

joblib.dump(model_rf,'../model/heart_model.pkl')