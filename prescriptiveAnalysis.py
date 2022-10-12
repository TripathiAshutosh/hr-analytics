import pandas as pd
import numpy as np

def singleAttributeBarPlotDataDict(singleAttributeData):
    keyValuePair = singleAttributeData.value_counts()
    dataDictionary = {}
    dataDictionary['categories'] = keyValuePair.index.tolist()
    dataDictionary['values'] = np.round(keyValuePair.astype(float)/keyValuePair.values.sum(),2).tolist()
    dataDictionary['text'] = ['{}%'.format(i) for i in np.round(keyValuePair.values.astype(float)/keyValuePair.values.sum(),4)*100]
    return dataDictionary

def histogramData(data):
    salaryDistribution = {}
    salaryDistribution['xData'] = data.values.tolist()
    return salaryDistribution


def groupedBarChart(data,groupList,columnsToGroup):
    temp = data.groupby(columnsToGroup).size().to_frame()
    temp = temp.reset_index()
    columnsToGroup.append('Count')
    temp.columns = columnsToGroup

    overtimeVsPerformanceRating = {}
    count = 0
    for count, group in enumerate(groupList,start=1):
        overtimeVsPerformanceRating['xData'+str(count)] = temp[columnsToGroup[0]][temp[columnsToGroup[1]]==group].values.tolist()
        overtimeVsPerformanceRating['yData'+str(count)] = np.round(temp[columnsToGroup[2]][temp[columnsToGroup[1]]==group]/temp[columnsToGroup[2]].values.sum(),2).tolist()
        overtimeVsPerformanceRating['text'+str(count)] = ['{}%'.format(i) for i in np.round(temp[columnsToGroup[2]][temp[columnsToGroup[1]]==group]/temp[columnsToGroup[2]].values.sum(),4)*100]
        overtimeVsPerformanceRating['name'+str(count)] = group

    return overtimeVsPerformanceRating
