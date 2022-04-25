from flask import Flask, render_template,request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('model/heart_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/data', methods=['post'])
def data():
    
    Mean_Age         = int(request.form.get('age'))
    Sex              = int(request.form.get('sex'))
    Race             = (request.form.get('race'))
    GenHealth        = int(request.form.get('GH'))
    BMI_cat          = (request.form.get('bmi'))
    Smoking	         = int(request.form.get('smoke'))
    SleepTime_cat    = (request.form.get('sleep'))
    DiffWalking      = int(request.form.get('DiffWalking'))
    PhysicalActivity = int(request.form.get('PhysicalActivity'))
    PhysicalHealth   = int(request.form.get('PhysicalHealth'))
    Diabetic         = (request.form.get('Diabetic'))
    Asthma           = int(request.form.get('Asthma'))
    KidneyDisease    = int(request.form.get('KidneyDisease'))
    SkinCancer       = int(request.form.get('SkinCancer'))
    Stroke           = int(request.form.get('Stroke'))

    
    Race_Asian=Race_Black=Race_Hispanic=Race_Other=Race_White=0
    
    if Race=='Asian':
        Race_Asian=1
    elif Race=='Black':
        Race_Black=1
    elif Race=='Hispanic':
        Race_Hispanic=1
    elif Race=='Other':
        Race_Other=1
    elif Race=='White':
        Race_White=1
    else:
        pass
    
    Diabetic_No_bd=Diabetic_Yes=Diabetic_Yes_preg=0
    
    if Diabetic=='N_bd':
        Diabetic_No_bd=1
    elif Diabetic=='Y':
        Diabetic_Yes=1
    elif Diabetic=='Y_preg':
        Diabetic_Yes_preg=1
    else:
        pass
    
    BMI_N=BMI_O=BMI_Owt=BMI_Uwt=0

    if BMI_cat=='N':
        BMI_N=1
    elif BMI_cat=='O':
        BMI_O=1
    elif BMI_cat=='Owt':
        BMI_Owt=1
    elif BMI_cat=='Uwt':
        BMI_Uwt=1
    else:
        pass
    
    SleepTime_Low=SleepTime_Normal=0
    
    if SleepTime_cat=='Low':
        SleepTime_Low=1
    elif SleepTime_cat=='Normal':
        SleepTime_Normal=1
    else:
        pass

    df = pd.DataFrame({
    "Smoking": [Smoking],
    "Stroke": [Stroke],
    "PhysicalHealth": [PhysicalHealth],
    "DiffWalking": [DiffWalking],
    "Sex": [Sex],
    "PhysicalActivity": [PhysicalActivity],
    "GenHealth": [GenHealth],
    "Asthma": [Asthma],
    "KidneyDisease": [KidneyDisease],
    "SkinCancer": [SkinCancer],
    "Mean_Age": [Mean_Age],
    
    "Race_Asian": [Race_Asian],
    "Race_Black": [Race_Black],
    "Race_Hispanic": [Race_Hispanic],
    "Race_Other": [Race_Other],
    "Race_White": [Race_White],
    
    "Diabetic_No, borderline diabetes": [Diabetic_No_bd],    
    "Diabetic_Yes": [Diabetic_Yes],
    'Diabetic_Yes (during pregnancy)': [Diabetic_Yes_preg],
    
    "BMI_N": [BMI_N],    
    "BMI_O": [BMI_O],
    "BMI_Owt": [BMI_Owt],
    "BMI_Uwt": [BMI_Uwt],
    
    "SleepTime_Low": [SleepTime_Low],
    "SleepTime_Normal": [SleepTime_Normal]
            
     })
    
    
    col=df.columns
    test=scaler.transform(df)    

    X_test = pd.DataFrame(test, columns = col)    

    result = model.predict(X_test)

    if result[0]==1:
        data = 'You have risk of Heart disease'
    else:
        data = 'Your heart seems healthy'

    print(data)

    return render_template('predict.html', data=data)


# app.run(host='0.0.0.0', port=8080)
app.run()

