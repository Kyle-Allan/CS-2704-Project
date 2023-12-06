import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sb
from scipy.stats import shapiro, spearmanr
import numpy as np
import statsmodels.api as sm

# before looking at levels of education


def mergeDatasets(formal, happy):
    # merging datasets by country names
    mergedDataset = pd.merge(formal, happy, left_on=['Entity', 'Year'], right_on=['Country', 'Year'], how='inner')
    return mergedDataset


def cleanEducationLevelDf(newDf):
    return newDf.drop(columns=['Code'])


def mergeLevelOfEducationDatasets(first, second):
    mergedEducation = pd.merge(first, second, on=['Entity', 'Year'], how='inner')
    return mergedEducation


def filterYearsEducationLevelDf(df):
    filteredDf = df[(df['Year'] == 2015) | (df['Year'] == 2020)]
    return filteredDf


def mergeEducationLevelsWithHappiness(education, happy):
    # merging datasets by country names
    mergedDataset = pd.merge(education, happy, left_on=['Entity', 'Year'], right_on=['Country', 'Year'], how='inner')
    return mergedDataset



def renameEducationLevelDf(df):
    df.rename(columns={'School enrollment, primary (% gross)': 'primary enrollment'}, inplace=True)
    df.rename(columns={'School enrollment, secondary (% gross)': 'secondary enrollment'}, inplace=True)
    df.rename(columns={'School enrollment, tertiary (% gross)': 'tertiary enrollment'}, inplace=True)
    return df


def createPivotedForNewDf(merged):
    # Pivot the DataFrame
    data = merged.pivot(index='Entity', columns='Year',
                              values=['Index', 'Rank', 'primary enrollment', 'secondary enrollment', 'tertiary enrollment'])
    # Flatten the multi-level columns
    data.columns = ['{}_{}'.format(col[1], col[0]) for col in data.columns]
    # Reset the index to make 'Entity' a column again
    data.reset_index(inplace=True)
    return data



def cleanDataframe(df):
    df = df.drop(columns=['Country'])
    df = df.drop(columns=['Code'])
    df = df.drop(columns=['Share of population with no formal education, 1820-2020'])
    return df


def createHeatmap(columns, dataFrame):
    correlation_matrix = dataFrame[columns].corr()
    # creating heatmap of the correlation matrix above
    sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    mp.title('Correlation Heatmap')
    mp.show()


def createYearDifferenceDataColumns(sample):
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


def cleanDf(cleanedDataframe):
    cleanedDataframe.rename(columns={'2015_Share of population with some formal education, 1820-2020': '2015_Share of population with some formal education'}, inplace=True)
    cleanedDataframe.rename(columns={'2020_Share of population with some formal education, 1820-2020': '2020_Share of population with some formal education'}, inplace=True)
    cleanedUpDataFrame = cleanedDataframe.dropna()
    return cleanedUpDataFrame


pd.set_option('display.max_columns', None)


# loading new datasets
enrollmentPrimary = pd.read_excel(r'C:\Users\kylea\OneDrive\2023 Fall Semester\Python\Datasets\PrimaryEducationEnrollment.xlsx')
enrollmentSecondary = pd.read_excel(r'C:\Users\kylea\OneDrive\2023 Fall Semester\Python\Datasets\SecondaryEducationEnrollment.xlsx')
enrollmentTertiary = pd.read_excel(r'C:\Users\kylea\OneDrive\2023 Fall Semester\Python\Datasets\TertiaryEducationEnrollment.xlsx')

enrollmentPrimary = cleanEducationLevelDf(enrollmentPrimary)
enrollmentSecondary = cleanEducationLevelDf(enrollmentSecondary)
enrollmentTertiary = cleanEducationLevelDf(enrollmentTertiary)

threeLevelOfEducationDf = mergeLevelOfEducationDatasets(enrollmentPrimary, enrollmentSecondary)
threeLevelOfEducationDf = mergeLevelOfEducationDatasets(threeLevelOfEducationDf, enrollmentTertiary)
threeLevelOfEducationDf = filterYearsEducationLevelDf(threeLevelOfEducationDf)


# loading datasets
formalEducation = pd.read_excel(r'C:\Users\kylea\OneDrive\2023 Fall Semester\Python\Datasets\Education & Happiness\Formal Education.xlsx')
happinessIndex = pd.read_excel(r'C:\Users\kylea\OneDrive\2023 Fall Semester\Python\Datasets\Education & Happiness\World Happiness Index by Reports 2013-2023.xlsx')


merged = mergeEducationLevelsWithHappiness(threeLevelOfEducationDf, happinessIndex)
pleaseWork = merged.drop(columns=['Country'])
pleaseWork = renameEducationLevelDf(pleaseWork)
pleaseWork = createPivotedForNewDf(pleaseWork)
pleaseWork = pleaseWork.dropna()
print(pleaseWork)


def shapiroTest(data):
    statistic, p_value = shapiro(data)
    print("Test Statistic:", statistic, "P-Value:", p_value)
    if p_value < 0.05:
        print('Data is not normally distributed')
    else:
        print('Data is normally distributed')


def spearmanRankTest(index, education):
    coefficient, p_value = spearmanr(index, education)
    print('coefficient:', coefficient, 'P-value:', p_value)

    if 0 < coefficient < 0.4:
        print('Weak relationship')
    elif 0.39 < coefficient < 0.6:
        print('Moderate relationship')
    elif 0.59 < coefficient < 0.8:
        print('Strong relationship')
    elif 0.79 < coefficient < 1:
        print('Very strong relationship')

    if p_value < 0.05:
        print('There is statistical significant correlation')
    else:
        print('There is not a statistical significant correlation')

print()
shapiroTest(pleaseWork['2015_Index'])
shapiroTest(pleaseWork['2020_Index'])

shapiroTest(pleaseWork['2015_secondary enrollment'])
shapiroTest(pleaseWork['2020_secondary enrollment'])
shapiroTest(pleaseWork['2015_tertiary enrollment'])
shapiroTest(pleaseWork['2020_tertiary enrollment'])
print()

spearmanRankTest(pleaseWork['2015_Index'], pleaseWork['2015_secondary enrollment'])
spearmanRankTest(pleaseWork['2015_Index'], pleaseWork['2015_tertiary enrollment'])
print()
spearmanRankTest(pleaseWork['2020_Index'], pleaseWork['2020_secondary enrollment'])
spearmanRankTest(pleaseWork['2020_Index'], pleaseWork['2020_tertiary enrollment'])


# making scatterplot
fig, (ax1, ax2) = mp.subplots(1, 2, figsize=(12, 6))

sb.regplot(x='2015_secondary enrollment', y='2015_Index', data=pleaseWork, color='red', ax=ax1)
sb.regplot(x='2015_tertiary enrollment', y='2015_Index', data=pleaseWork, color='blue', ax=ax2)
mp.title('2015')
mp.show()

# calculating slope of the line of best fit for tertiary and secondary in 2015
x_with_constant = sm.add_constant(pleaseWork[['2015_secondary enrollment', '2015_tertiary enrollment']])
# Fit the linear regression model
model = sm.OLS(pleaseWork['2015_Index'], x_with_constant).fit()
# Get the slope (beta1) and (beta2) from the model summary
slope1 = model.params.iloc[1]
slope2 = model.params.iloc[2]
print("Slope (beta1):", slope1, 'Slope (beta2:)', slope2)


fig, (ax1, ax2) = mp.subplots(1, 2, figsize=(12, 6))

sb.regplot(x='2020_secondary enrollment', y='2020_Index', data=pleaseWork, color='red', ax=ax1)
sb.regplot(x='2020_tertiary enrollment', y='2020_Index', data=pleaseWork, color='blue', ax=ax2)
mp.title('2020')
mp.show()



# calculating slope of the line of best fit for tertiary and secondary in 2020
x_with_constant = sm.add_constant(pleaseWork[['2020_secondary enrollment', '2020_tertiary enrollment']])
# Fit the linear regression model
model = sm.OLS(pleaseWork['2020_Index'], x_with_constant).fit()
# Get the slope (beta1) and (beta2) from the model summary
slope1 = model.params.iloc[1]
slope2 = model.params.iloc[2]
print("Slope (beta1):", slope1, 'Slope (beta2:)', slope2)


#histograms of data for 2015
fig, axes = mp.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Plot histogram for Happiness Index 2015
axes[0, 0].hist(pleaseWork['2015_Index'], bins='auto', edgecolor='black', color='skyblue')
axes[0, 0].set_title('Happiness Index in 2015')
axes[0, 0].set_xlabel('Happiness Index')
axes[0, 0].set_ylabel('Frequency')

# Plot histogram for Happiness Index 2015
axes[0, 1].hist(pleaseWork['2015_secondary enrollment'], bins='auto', edgecolor='black', color='lightcoral')
axes[0, 1].set_title('Secondary enrollment in 2015')
axes[0, 1].set_xlabel('Secondary enrollment')
axes[0, 1].set_ylabel('Frequency')

axes[1, 0].axis('off')

# Plot histogram for Happiness Index 2015
axes[1, 1].hist(pleaseWork['2015_tertiary enrollment'], bins='auto', edgecolor='black', color='lightcoral')
axes[1, 1].set_title('Tertiary enrollment in 2015')
axes[1, 1].set_xlabel('Tertiary enrollment')
axes[1, 1].set_ylabel('Frequency')
mp.tight_layout()
mp.show()

#histograms of data for 2020
fig, axes = mp.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Plot histogram for Happiness Index 2015
axes[0, 0].hist(pleaseWork['2020_Index'], bins='auto', edgecolor='black', color='skyblue')
axes[0, 0].set_title('Happiness Index in 2020')
axes[0, 0].set_xlabel('Happiness Index')
axes[0, 0].set_ylabel('Frequency')

# Plot histogram for Happiness Index 2020
axes[0, 1].hist(pleaseWork['2020_secondary enrollment'], bins='auto', edgecolor='black', color='lightcoral')
axes[0, 1].set_title('Secondary enrollment in 2020')
axes[0, 1].set_xlabel('Secondary enrollment')
axes[0, 1].set_ylabel('Frequency')

axes[1, 0].axis('off')

# Plot histogram for Happiness Index 2020
axes[1, 1].hist(pleaseWork['2020_tertiary enrollment'], bins='auto', edgecolor='black', color='lightcoral')
axes[1, 1].set_title('Tertiary enrollment in 2020')
axes[1, 1].set_xlabel('Tertiary enrollment')
axes[1, 1].set_ylabel('Frequency')
mp.tight_layout()
mp.show()





'''
# merging datasets
merged = mergeDatasets(formalEducation, happinessIndex)
merged = cleanDataframe(merged)


# creating dataframe with all relevant info on entity in one row
cleanedDataframe = createPivotedDf(merged)
cleanedDataframe = cleanDf(cleanedDataframe)
# adding extra calculated data to dataframe
createYearDifferenceDataColumns(cleanedDataframe)

print(cleanedDataframe)


# shapiro test of normality
statistic_happiness, p_value_happiness = shapiro(cleanedDataframe['2015_Index'])
statistic_education, p_value_education = shapiro(cleanedDataframe['2015_Share of population with some formal education'])
print(f"Shapiro-Wilk Test for Happiness Index - Statistic 2015: {statistic_happiness}, P-value: {p_value_happiness}")
print(f"Shapiro-Wilk Test for Education Levels - Statistic 2015: {statistic_education}, P-value: {p_value_education}")

statistic_happiness, p_value_happiness = shapiro(cleanedDataframe['2020_Index'])
statistic_education, p_value_education = shapiro(cleanedDataframe['2020_Share of population with some formal education'])
print(f"Shapiro-Wilk Test for Happiness Index - Statistic 2020: {statistic_happiness}, P-value: {p_value_happiness}")
print(f"Shapiro-Wilk Test for Education Levels - Statistic 2020: {statistic_education}, P-value: {p_value_education}")


# spearman rank test to test correlation
corr, pval = spearmanr(cleanedDataframe['2015_Index'], cleanedDataframe['2015_Share of population with some formal education'])
print("Spearman's rank correlation coefficient for 2015:", corr)
print("P-value for 2015:", pval)

corr, pval = spearmanr(cleanedDataframe['2020_Index'], cleanedDataframe['2020_Share of population with some formal education'])
print("Spearman's rank correlation coefficient for 2020:", corr)
print("P-value for 2020:", pval)


fig, axes = mp.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Plot histogram for Happiness Index 2015
axes[0, 0].hist(cleanedDataframe['2015_Index'], bins='auto', edgecolor='black', color='skyblue')
axes[0, 0].set_title('Happiness Index in 2015')
axes[0, 0].set_xlabel('Happiness Index')
axes[0, 0].set_ylabel('Frequency')

# Plot histogram for Education Levels 2015
axes[0, 1].hist(cleanedDataframe['2015_Share of population with some formal education'], bins='auto', edgecolor='black', color='lightcoral')
axes[0, 1].set_title('Education Levels in 2015')
axes[0, 1].set_xlabel('Education Levels')
axes[0, 1].set_ylabel('Frequency')

# Plot histogram for Happiness Index 2020
axes[1, 0].hist(cleanedDataframe['2020_Index'], bins='auto', edgecolor='black', color='skyblue')
axes[1, 0].set_title('Happiness Index in 2020')
axes[1, 0].set_xlabel('Happiness Index')
axes[1, 0].set_ylabel('Frequency')

# Plot histogram for Education Levels 2020
axes[1, 1].hist(cleanedDataframe['2020_Share of population with some formal education'], bins='auto', edgecolor='black', color='lightcoral')
axes[1, 1].set_title('Education Levels in 2020')
axes[1, 1].set_xlabel('Education Levels')
axes[1, 1].set_ylabel('Frequency')

mp.tight_layout()
mp.show()


def createAndDisplayBasicHappyStatistics(cleanedDataframe):
    stdDev_2015 = round(cleanedDataframe['2015_Index'].std(), 2)
    stdDev_2020 = round(cleanedDataframe['2020_Index'].std(), 2)
    minIndex_2015 = round(cleanedDataframe['2015_Index'].min(), 2)
    minIndex_2020 = round(cleanedDataframe['2020_Index'].min(), 2)
    maxIndex_2015 = round(cleanedDataframe['2015_Index'].max(), 2)
    maxIndex_2020 = round(cleanedDataframe['2020_Index'].max(), 2)
    meanIndex_2015 = round(cleanedDataframe['2015_Index'].mean(), 2)
    meanIndex_2020 = round(cleanedDataframe['2020_Index'].mean(), 2)
    medianIndex_2015 = round(cleanedDataframe['2015_Index'].median(), 2)
    medianIndex_2020 = round(cleanedDataframe['2020_Index'].median(), 2)
    sampleSize_2015 = cleanedDataframe['2015_Index'].count()
    sampleSize_2020 = cleanedDataframe['2020_Index'].count()


    fig, ax = mp.subplots()
    ax.axis('off')  # Turn off axis for table
    # Table data
    table_data = [['2015', sampleSize_2015, minIndex_2015, maxIndex_2015, meanIndex_2015, medianIndex_2015, stdDev_2015],
                  ['2020', sampleSize_2020, minIndex_2020, maxIndex_2020, meanIndex_2020, medianIndex_2020, stdDev_2020]]
    # Create a table
    table = ax.table(cellText=table_data, colLabels=['Year', 'Sample Size', 'Min Index', 'Max Index', 'Mean Index', 'Median Index',  'Standard Deviation'], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    table.auto_set_column_width([0, 1, 2, 3, 4, 5, 6])
    # Display the table
    mp.title('Happiness Statistics', fontsize=20)
    mp.show()


createAndDisplayBasicHappyStatistics(cleanedDataframe)


def createAndDisplayBasicEducationStatistics(cleanedDataframe):
    stdDev_2015 = round(cleanedDataframe['2015_Share of population with some formal education'].std(), 2)
    stdDev_2020 = round(cleanedDataframe['2020_Share of population with some formal education'].std(), 2)
    minEducation_2015 = round(cleanedDataframe['2015_Share of population with some formal education'].min(), 2)
    minEducation_2020 = round(cleanedDataframe['2020_Share of population with some formal education'].min(), 2)
    maxEducation_2015 = round(cleanedDataframe['2015_Share of population with some formal education'].max(), 2)
    maxEducation_2020 = round(cleanedDataframe['2020_Share of population with some formal education'].max(), 2)
    meanEducation_2015 = round(cleanedDataframe['2015_Share of population with some formal education'].mean(), 2)
    meanEducation_2020 = round(cleanedDataframe['2020_Share of population with some formal education'].mean(), 2)
    medianEducation_2015 = round(cleanedDataframe['2015_Share of population with some formal education'].median(), 2)
    medianEducation_2020 = round(cleanedDataframe['2020_Share of population with some formal education'].median(), 2)
    sampleSize_2015 = cleanedDataframe['2015_Share of population with some formal education'].count()
    sampleSize_2020 = cleanedDataframe['2020_Share of population with some formal education'].count()


    fig, ax = mp.subplots()
    ax.axis('off')  # Turn off axis for table
    # Table data
    table_data = [['2015', sampleSize_2015, minEducation_2015, maxEducation_2015, meanEducation_2015, medianEducation_2015, stdDev_2015],
                  ['2020', sampleSize_2020, minEducation_2020, maxEducation_2020, meanEducation_2020, medianEducation_2020, stdDev_2020]]
    # Create a table
    table = ax.table(cellText=table_data,
                     colLabels=['Year', 'Sample Size', 'Min Education Percent', 'Max Education Percent', 'Mean Education Percent', 'Median Education Percent', 'Standard Deviation'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    table.auto_set_column_width([0, 1, 2, 3, 4, 5, 6])
    # Display the table
    mp.title('Education Statistics', fontsize=20)
    mp.show()


createAndDisplayBasicEducationStatistics(cleanedDataframe)


# making correlation matrix on the index column and population with formal education
columnsOfInterest = ['Share of population with some formal education, 1820-2020', 'Index']
# heatmap on columns above
createHeatmap(columnsOfInterest, merged)


sb.lmplot(x='Share of population with some formal education, 1820-2020', y='Index', hue='Year', data=merged, scatter_kws={'s': 50}, height=6, aspect=1.5)
mp.xlabel('Share of population with some formal education 2015/2020')
mp.ylabel('Happiness Index')
mp.show()
'''
