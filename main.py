import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sb
import numpy as np
from scipy.stats import pearsonr


def mergeDatasets(formal, happy):
    # merging datasets by country names
    mergedDataset = pd.merge(formal, happy, left_on=['Entity', 'Year'], right_on=['Country', 'Year'], how='inner')
    return mergedDataset


def dfSpecificYearOnly(dataframe, year):
    dataframe = dataframe[dataframe['Year'] == year]
    return dataframe


def cleanDataframe(df):
    df = df.drop(columns=['Country'])
    df = df.drop(columns=['Code'])
    df = df.drop(columns=['Share of population with no formal education, 1820-2020'])
    return df


def printMerged(mergedDF):
    print(mergedDF)


def createHeatmap(columns, dataFrame):
    correlation_matrix = dataFrame[columns].corr()
    # creating heatmap of the correlation matrix above
    sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    mp.title('Correlation Heatmap')
    mp.show()


def pearsonCorrelation(educationColumn, happyIndex):
    return pearsonr(educationColumn, happyIndex)


def printBasicHappyStatisticsInYear(dataframe, year):
    dataframe = dataframe[dataframe['Year'] == year]
    print('Mean happiness index in year', year, ' is ', np.nanmean(dataframe['Index'].tolist()))


# datasets
formalEducation = pd.read_excel(r'C:\Users\kylea\OneDrive\2023 Fall Semester\Python\Datasets\Education & Happiness\Formal Education.xlsx')
happinessIndex = pd.read_excel(r'C:\Users\kylea\OneDrive\2023 Fall Semester\Python\Datasets\Education & Happiness\World Happiness Index by Reports 2013-2023.xlsx')
pd.set_option('display.max_columns', None)

'''
# dataframe with only values in year 2020
happy2020 = dfSpecificYearOnly(happinessIndex, 2020)
formal2020 = dfSpecificYearOnly(formalEducation, 2020)

print(happy2020)
'''

# merging datasets
merged = mergeDatasets(formalEducation, happinessIndex)
merged = cleanDataframe(merged)
print(merged)


merged2020 = dfSpecificYearOnly(merged, 2020)
print(merged2020)

merged2015 = dfSpecificYearOnly(merged, 2015)
print(merged2015)

printBasicHappyStatisticsInYear(merged, 2015)
printBasicHappyStatisticsInYear(merged, 2020)


'''
# creating list of education percent and happiness index where value of year is 2020 so I can do a p value on it
education2020Column = merged2020['Share of population with some formal education, 1820-2020'].tolist()
happyindex2020Column = merged2020['Index'].tolist()
print(pearsonCorrelation(education2020Column, happyindex2020Column))

printBasicHappyStatisticsInYear(happinessIndex, 2020)

'''
# making correlation matrix on the index column and population with formal education
columnsOfInterest = ['Share of population with some formal education, 1820-2020', 'Index']
# heatmap on columns above
createHeatmap(columnsOfInterest, merged)

