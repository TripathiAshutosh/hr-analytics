from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd
import prescriptiveAnalysis as pa
import predictiveAnalysis as pred
app = Flask(__name__)

@app.route('/',methods=["GET","POST"])
def index():
    return render_template("Dashboards/dashboard.html")


@app.route('/demo-models',methods=["Get","POST"])
def demoModels():
    return render_template("demo-models.html")


@app.route('/getResponseLinearReg',methods=["GET","POST"])
def getResponseLinearReg():
    print("inside python")
    CRIM = np.array(request.form["CRIM"], dtype=float)
    ZN = np.array(request.form["ZN"], dtype=float)
    INDUS = np.array(request.form["INDUS"], dtype=float)
    CHAS = np.array(request.form["CHAS"], dtype=float)
    NOX = np.array(request.form["NOX"], dtype=float)
    RM = np.array(request.form["RM"], dtype=float)
    AGE = np.array(request.form["AGE"], dtype=float)
    DIS = np.array(request.form["DIS"], dtype=float)
    RAD = np.array(request.form["RAD"], dtype=float)
    TAX = np.array(request.form["TAX"], dtype=float)
    PT = np.array(request.form["PT"], dtype=float)
    B = np.array(request.form["B"], dtype=float)
    LSTAT = np.array(request.form["LSTAT"], dtype=float)
    inputList = [CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PT,B,LSTAT]
    with open("boston_mlm.pkl", 'rb') as file:
            pickle_model = pickle.load(file)
            y_pred_from_pkl = pickle_model.predict([inputList])
    print(y_pred_from_pkl)
    return str(y_pred_from_pkl[0])



@app.route('/contact',methods=["Get","POST"])
def contact():
    return render_template("contact.html")



def preprocessing():
    data = pd.read_csv("HR_Attrition.csv")
    Education_dict = {1:'Below College',
           2:'College',
           3:'Bachelor',
           4:'Master',
           5:'Doctor',
           }
    EnvironmentSatisfaction_dict = {1:'Low',
           2:'Medium',
           3:'High',
           4:'Very High',
           }

    JobInvolvement_dict = {1:'Low',
           2:'Medium',
           3:'High',
           4:'Very High',
           }

    JobSatisfaction_dict = {1:'Low',
           2:'Medium',
           3:'High',
           4:'Very High',
           }

    PerformanceRating_dict = {1:'Low',
           2:'Good',
           3:'Excellent',
           4:'Outstanding',
           }

    RelationshipSatisfaction_dict = {1:'Low',
           2:'Medium',
           3:'High',
           4:'Very High',
           }

    WorkLifeBalance_dict = {1:'Bad',
           2:'Good',
           3:'Better',
           4:'Best',
           }
    data = data.replace({"Education":Education_dict,
                "EnvironmentSatisfaction":EnvironmentSatisfaction_dict,
                "JobInvolvement":JobInvolvement_dict,
                "JobSatisfaction":JobSatisfaction_dict,
                 
                "PerformanceRating":PerformanceRating_dict,
                 
                "RelationshipSatisfaction":RelationshipSatisfaction_dict,
                
                "WorkLifeBalance":WorkLifeBalance_dict,
                 
                })
    cat_cols = []
    for i in data.columns:
        if data[i].dtype =='object' or len(np.unique(data[i]))<=15 : # if the number of levels is less that 15 considering the column as categorial
            cat_cols.append(i)
            #print("{} : {} : {} ".format(i,len(np.unique(data[i])),np.unique(data[i])))
    for i in cat_cols:
        data[i] = data[i].astype('category')
    
    num_cols = [i for i in data.columns if i not in cat_cols]

    return num_cols, cat_cols, data
    


@app.route('/dashboard',methods=["Get","POST"])
def dashboard():
    '''HR Analytics Dashboard - Prescriptive and Predictive analytics
    '''
    
    return render_template("Dashboards/dashboard.html")

@app.route('/pres-dashboard',methods=["Get","POST"])
def presDashboard():
    '''HR Analytics - Prescriptive Dashboard
    '''
    num_cols, cat_cols, data = preprocessing()
    attrition = data.Attrition.value_counts()
    
    attrDictionary = pa.singleAttributeBarPlotDataDict(data.Attrition)
    genderDsitribution = pa.singleAttributeBarPlotDataDict(data.Gender)
    businessTravelDict = pa.singleAttributeBarPlotDataDict(data.BusinessTravel)
    deptWiseEmpDist = pa.singleAttributeBarPlotDataDict(data.Department)
    skillDistibution = pa.singleAttributeBarPlotDataDict(data.EducationField)
    jobRoleDistibution = pa.singleAttributeBarPlotDataDict(data.JobRole)
    ageDistibution = pa.singleAttributeBarPlotDataDict(data.Age)
    salaryDistribution = pa.histogramData(data.MonthlyIncome)

    performanceRatings = ["Excellent","Outstanding"]
    columnsToGroup = ['OverTime','PerformanceRating']
    overTimeAndPerformanceRating = pa.groupedBarChart(data,performanceRatings,columnsToGroup)

    jobSatisfactionLevels = ["Low","Medium","High","Very High"]
    columnsToGroup = ['YearsWithCurrManager','JobSatisfaction']
    jobSatVsYrWithMngr = pa.groupedBarChart(data,jobSatisfactionLevels,columnsToGroup)

    attritionLevels = ["Yes","No"]
    columnsToGroup = ['Gender','Attrition']
    attritionAndGender = pa.groupedBarChart(data,attritionLevels,columnsToGroup)

    attritionLevels = ["Yes","No"]
    columnsToGroup = ['BusinessTravel','Attrition']
    attritionAndBusTravel = pa.groupedBarChart(data,attritionLevels,columnsToGroup)

    attritionLevels = ["Yes","No"]
    columnsToGroup = ['Department','Attrition']
    attritionAndDepartment = pa.groupedBarChart(data,attritionLevels,columnsToGroup)

    attritionLevels = ["Yes","No"]
    columnsToGroup = ['MaritalStatus','Attrition']
    attritionAndMaritalStatus = pa.groupedBarChart(data,attritionLevels,columnsToGroup)

    attritionLevels = ["Yes","No"]
    columnsToGroup = ['WorkLifeBalance','Attrition']
    attritionAndWorkLifeBlnc = pa.groupedBarChart(data,attritionLevels,columnsToGroup)

    attritionLevels = ["Yes","No"]
    columnsToGroup = ['TotalWorkingYears','Attrition']
    attritionAndExperience = pa.groupedBarChart(data,attritionLevels,columnsToGroup)

    return render_template("Dashboards/pres-dashboard.html", attrDictionary = attrDictionary,
    genderDsitribution=genderDsitribution,businessTravelDict=businessTravelDict,
    deptWiseEmpDist=deptWiseEmpDist,skillDistibution=skillDistibution,jobRoleDistibution=jobRoleDistibution,
    ageDistibution=ageDistibution,salaryDistribution=salaryDistribution,
    overTimeAndPerformanceRating=overTimeAndPerformanceRating,jobSatVsYrWithMngr=jobSatVsYrWithMngr,
    attritionAndGender=attritionAndGender,attritionAndBusTravel=attritionAndBusTravel,
    attritionAndDepartment=attritionAndDepartment,attritionAndMaritalStatus=attritionAndMaritalStatus,
    attritionAndWorkLifeBlnc=attritionAndWorkLifeBlnc,attritionAndExperience=attritionAndExperience)

@app.route('/pred-dashboard',methods=["Get","POST"])
def predDashboard():
    '''HR Analytics - Predictive Dasboard
    '''
    return render_template("Dashboards/pred-dashboard.html")


@app.route('/predictUsingXGB',methods=["Get","POST"])
def predictUsingXGB():
    import joblib
    import pandas as pd
  
    uploaded_file = request.files['file']
    df = pd.read_csv(uploaded_file)
    
    print(df.head())
    X_test = pred.prepareX_test(df)

    with open("model/xgb_clf.pkl", 'rb') as file:
            xgb_clf = joblib.load(file)
    #xgb_clf = joblib.load('xgb_clf.pkl')
    predictions_test= xgb_clf.predict(X_test)
    print(predictions_test)
    predictedProbailityScoresForEachClass = xgb_clf.predict_proba(X_test)
    prob = pd.DataFrame(predictedProbailityScoresForEachClass)
    prob.head()
    result = pd.DataFrame()
    result['Emp_ID']=X_test.index
    result['Attrition'] = predictions_test
    result['prob_being_Churnig'] = predictedProbailityScoresForEachClass[:,1]
    result['prob_being_Staying'] = predictedProbailityScoresForEachClass[:,0]
    result['Attrition'] = result['Attrition'].apply(lambda x: 'Yes' if x == 1 else 'No')
    #return result
    #return render_template('Dashboards/df.html',  tables=[result.to_html(classes='data')], titles=result.columns.values)
    #return render_template('Dashboards/df.html',  tables=[result.to_html(classes='data', header="true")])
    return render_template("Dashboards/df.html", column_names=result.columns.values, row_data=list(result.values.tolist()))

def predictEmpChurnProbUsingXGB():
    import joblib
    X_test = pred.prepareX_test(df)
    xgb_clf = joblib.load('model/xgb_clf.pkl')
    xgb_clf.predict(X_test)
    return X_test

if __name__ == '__main__':
    app.run(debug=True)
