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


def printBasicStatisticsInYear(dataframe, year):
    dataframe = dataframe[dataframe['Year'] == year]
    print('Mean happiness index in year', year, ' is ', np.nanmean(dataframe['Index'].tolist()))
    print('Min happiness index in year', year, ' is ', dataframe['Index'].min())
    print('Max happiness index in year', year, ' is ', dataframe['Index'].max())
    print('Std Dev happiness index in year', year, ' is ', dataframe['Index'].std())
    print('Mean share of population with some formal education in year', year, ' is ',
          np.nanmean(dataframe['Share of population with some formal education, 1820-2020'].tolist()))
    print('Min share of population with some formal education in year', year, ' is ', dataframe['Share of population with some formal education, 1820-2020'].min())
    print('Max share of population with some formal education in year', year, ' is ', dataframe['Share of population with some formal education, 1820-2020'].max())
    print('Std Dev share of population with some formal education in year', year, ' is ', dataframe['Share of population with some formal education, 1820-2020'].std())


# datasets
formalEducation = pd.read_excel(r'C:\Users\kylea\OneDrive\2023 Fall Semester\Python\Datasets\Education & Happiness\Formal Education.xlsx')
happinessIndex = pd.read_excel(r'C:\Users\kylea\OneDrive\2023 Fall Semester\Python\Datasets\Education & Happiness\World Happiness Index by Reports 2013-2023.xlsx')
pd.set_option('display.max_columns', None)


# merging datasets
merged = mergeDatasets(formalEducation, happinessIndex)
merged = cleanDataframe(merged)
print(merged)


printBasicStatisticsInYear(merged, 2015)
print()
printBasicStatisticsInYear(merged, 2020)


# crea
# ting list of education percent and happiness index where value of year is 2020 so I can do a p value on it
formalEducationColumn = merged['Share of population with some formal education, 1820-2020'].tolist()
happyindexColumn = merged['Index'].tolist()
# print(pearsonCorrelation(formalEducationColumn, happyindexColumn))


# making correlation matrix on the index column and population with formal education
columnsOfInterest = ['Share of population with some formal education, 1820-2020', 'Index']
# heatmap on columns above
createHeatmap(columnsOfInterest, merged)


sb.lmplot(x='Share of population with some formal education, 1820-2020', y='Index', hue='Year', data=merged, scatter_kws={'s': 50}, height=6, aspect=1.5)
mp.xlabel('Share of population with some formal education 2015/2020')
mp.ylabel('Happiness Index')
mp.show()
