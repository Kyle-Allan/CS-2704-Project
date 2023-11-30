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


def createIndexDataColumns(sample):
    sample['Index Difference'] = sample['2020_Index'] - sample['2015_Index']
    sample['Index Percent Change'] = ((sample['2020_Index'] - sample['2015_Index']) / sample['2015_Index']) * 100
    sample['Education Difference '] = sample['2020_Share of population with some formal education'] - sample['2015_Share of population with some formal education']
    sample['Education Percent Change'] = ((sample['2020_Share of population with some formal education'] - sample['2015_Share of population with some formal education']) / sample['2015_Share of population with some formal education']) * 100


def createPivotedDf(merged):
    # Pivot the DataFrame
    data = merged.pivot(index='Entity', columns='Year',
                              values=['Index', 'Rank', 'Share of population with some formal education, 1820-2020'])
    # Flatten the multi-level columns
    data.columns = ['{}_{}'.format(col[1], col[0]) for col in data.columns]
    # Reset the index to make 'Entity' a column again
    data.reset_index(inplace=True)
    return data


def cleanPivotedDf(pivoted_df):
    pivoted_df.rename(columns={'2015_Share of population with some formal education, 1820-2020': '2015_Share of population with some formal education'}, inplace=True)
    pivoted_df.rename(columns={'2020_Share of population with some formal education, 1820-2020': '2020_Share of population with some formal education'}, inplace=True)
    cleanedUpDataFrame = pivoted_df.dropna()
    return cleanedUpDataFrame


# datasets
formalEducation = pd.read_excel(r'C:\Users\kylea\OneDrive\2023 Fall Semester\Python\Datasets\Education & Happiness\Formal Education.xlsx')
happinessIndex = pd.read_excel(r'C:\Users\kylea\OneDrive\2023 Fall Semester\Python\Datasets\Education & Happiness\World Happiness Index by Reports 2013-2023.xlsx')
pd.set_option('display.max_columns', None)

# merging datasets
merged = mergeDatasets(formalEducation, happinessIndex)
merged = cleanDataframe(merged)
print(merged)


# creating dataframe with all relevant info on entity in one row
pivoted_df = createPivotedDf(merged)
pivoted_df = cleanPivotedDf(pivoted_df)
# adding extra calculated data to dataframe
createIndexDataColumns(pivoted_df)
print(pivoted_df)

print(pearsonCorrelation(pivoted_df['2015_Share of population with some formal education'], pivoted_df['2015_Index']))


def createAndDisplayBasicHappyStatistics(pivoted_df):
    stdDev_2015 = round(pivoted_df['2015_Index'].std(), 2)
    stdDev_2020 = round(pivoted_df['2020_Index'].std(), 2)
    minIndex_2015 = round(pivoted_df['2015_Index'].min(), 2)
    minIndex_2020 = round(pivoted_df['2020_Index'].min(), 2)
    maxIndex_2015 = round(pivoted_df['2015_Index'].max(), 2)
    maxIndex_2020 = round(pivoted_df['2020_Index'].max(), 2)
    meanIndex_2015 = round(pivoted_df['2015_Index'].mean(), 2)
    meanIndex_2020 = round(pivoted_df['2020_Index'].mean(), 2)


    fig, ax = mp.subplots()
    ax.axis('off')  # Turn off axis for table
    # Table data
    table_data = [['2015', minIndex_2015, maxIndex_2015, meanIndex_2015, stdDev_2015],
                  ['2020', minIndex_2020, maxIndex_2020, meanIndex_2020, stdDev_2020]]
    # Create a table
    table = ax.table(cellText=table_data, colLabels=['Year', 'Min Index', 'Max Index', 'Mean Index', 'Standard Deviation'], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    table.auto_set_column_width([0, 1, 2, 3, 4])
    # Display the table
    mp.title('Happiness Statistics', fontsize=20)
    mp.show()


createAndDisplayBasicHappyStatistics(pivoted_df)


def createAndDisplayBasicEducationStatistics(pivoted_df):
    stdDev_2015 = round(pivoted_df['2015_Share of population with some formal education'].std(), 2)
    stdDev_2020 = round(pivoted_df['2020_Share of population with some formal education'].std(), 2)
    minEducation_2015 = round(pivoted_df['2015_Share of population with some formal education'].min(), 2)
    minEducation_2020 = round(pivoted_df['2020_Share of population with some formal education'].min(), 2)
    maxEducation_2015 = round(pivoted_df['2015_Share of population with some formal education'].max(), 2)
    maxEducation_2020 = round(pivoted_df['2020_Share of population with some formal education'].max(), 2)
    meanEducation_2015 = round(pivoted_df['2015_Share of population with some formal education'].mean(), 2)
    meanEducation_2020 = round(pivoted_df['2020_Share of population with some formal education'].mean(), 2)


    fig, ax = mp.subplots()
    ax.axis('off')  # Turn off axis for table
    # Table data
    table_data = [['2015', minEducation_2015, maxEducation_2015, meanEducation_2015, stdDev_2015],
                  ['2020', minEducation_2020, maxEducation_2020, meanEducation_2020, stdDev_2020]]
    # Create a table
    table = ax.table(cellText=table_data,
                     colLabels=['Year', 'Min Education Percent', 'Max Education Percent', 'Mean Education Percent', 'Standard Deviation'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    table.auto_set_column_width([0, 1, 2, 3, 4])
    # Display the table
    mp.title('Education Statistics', fontsize=20)
    mp.show()


createAndDisplayBasicEducationStatistics(pivoted_df)


# making correlation matrix on the index column and population with formal education
columnsOfInterest = ['Share of population with some formal education, 1820-2020', 'Index']
# heatmap on columns above
createHeatmap(columnsOfInterest, merged)


sb.lmplot(x='Share of population with some formal education, 1820-2020', y='Index', hue='Year', data=merged, scatter_kws={'s': 50}, height=6, aspect=1.5)
mp.xlabel('Share of population with some formal education 2015/2020')
mp.ylabel('Happiness Index')
mp.show()
