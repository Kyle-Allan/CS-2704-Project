import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sb

formalEducation = pd.read_excel(r'C:\Users\kylea\OneDrive\2023 Fall Semester\Python\Datasets\Education & Happiness\Formal Education.xlsx')
happinessIndex = pd.read_excel(r'C:\Users\kylea\OneDrive\2023 Fall Semester\Python\Datasets\Education & Happiness\World Happiness Index by Reports 2013-2023.xlsx')


# printing excel files
print(formalEducation)
print(happinessIndex)

happy2020 = happinessIndex[happinessIndex['Year'] == 2020]
formal2020 = formalEducation[formalEducation['Year'] == 2020]

print(happy2020)
print(formal2020)

merged = pd.merge(formal2020, happy2020, left_on='Entity', right_on='Country', how='inner')
merged = merged.drop(columns=['Country'])
merged = merged.drop_duplicates(subset=['Entity'])

pd.set_option('display.max_columns', None)
print(merged)
