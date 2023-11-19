import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sb
from scipy.stats import pearsonr


def mergeDatasets(formal, happy):
    # merging datasets by country names and dropping unnecessary columns
    merged = pd.merge(formal, happy, left_on='Entity', right_on='Country', how='inner')
    return merged


def dropUnnecessaryColumns(merged):
    merged = merged.drop(columns=['Country'])
    merged = merged.drop(columns=['Code'])
    merged = merged.drop(columns=['Share of population with no formal education, 1820-2020'])
    merged = merged.drop(columns=['Year_y'])
    # removing duplicate country names
    merged = merged.drop_duplicates(subset=['Entity'])
    return merged


def printMerged(merged):
    print(merged)


def createHeatmap(columns, merged):
    correlation_matrix = merged[columns].corr()
    # creating heatmap of the correlation matrix above
    sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    mp.title('Correlation Heatmap 2020')
    mp.show()


def pearsonCorrelation(educationColumn, happyIndex):
    return pearsonr(educationColumn, happyIndex)



formalEducation = pd.read_excel(r'C:\Users\kylea\OneDrive\2023 Fall Semester\Python\Datasets\Education & Happiness\Formal Education.xlsx')
happinessIndex = pd.read_excel(r'C:\Users\kylea\OneDrive\2023 Fall Semester\Python\Datasets\Education & Happiness\World Happiness Index by Reports 2013-2023.xlsx')

# dataframe with only values in year 2020
happy2020 = happinessIndex[happinessIndex['Year'] == 2020]
formal2020 = formalEducation[formalEducation['Year'] == 2020]

pd.set_option('display.max_columns', None)

# merging datsets
merged2020 = mergeDatasets(formal2020, happy2020)

education2020Column = merged2020['Share of population with some formal education, 1820-2020'].tolist()
happyindex2020Column = merged2020['Index'].tolist()
print(pearsonCorrelation(education2020Column, happyindex2020Column))

# dropping unnecessary columns and duplicate entity values
merged2020 = dropUnnecessaryColumns(merged2020)
printMerged(merged2020)

# making correlation matrix on the index column and population with formal education
columns_of_interest = ['Share of population with some formal education, 1820-2020', 'Index']

createHeatmap(columns_of_interest, merged2020)


