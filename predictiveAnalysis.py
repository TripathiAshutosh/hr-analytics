import pandas as pd
def prepareX_test(df):
    dummy_col = ['BusinessTravel',
 'Department',
 'Education',
 'EducationField',
 'EnvironmentSatisfaction',
 'Gender',
 'JobInvolvement',
 'JobLevel',
 'JobRole',
 'JobSatisfaction',
 'MaritalStatus',
 'NumCompaniesWorked',
 'OverTime',
 'PercentSalaryHike',
 'PerformanceRating',
 'RelationshipSatisfaction',
 'StockOptionLevel',
 'TrainingTimesLastYear',
 'WorkLifeBalance',
 'YearsInCurrentRole',
 'YearsSinceLastPromotion',
 'YearsWithCurrManager']
    X_test = pd.get_dummies(df, columns=dummy_col, drop_first=True, dtype='uint8')
    #data.info()
    X_test = X_test.T.drop_duplicates()
    X_test = X_test.T

    # Remove Duplicate Rows
    X_test.drop_duplicates(inplace=True)

    print(X_test.shape)
    return X_test