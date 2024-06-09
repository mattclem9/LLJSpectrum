# %%
#importing pandas
import pandas as pd

# %%
#creating inital llj dataframe
dfllj = pd.read_csv('/Users/matt/Desktop/LLJ-Data/llj.csv')

# %%
#creating inital spec dataframe
dfspec = pd.read_csv('/Users/matt/Desktop/LLJ-Data/spectrum.csv')

# %%
#getting information about llj dataframe
dfllj.describe(include = 'all')

# %%
#determining number of observation missing author field llj
AuthorBlankLLJ = dfllj['Author'].isna().sum()
print(f"Number of rows missing the '{'Author'}' field in '{'dfllj'}' {AuthorBlankLLJ}")

# %%
#determining number of observations missing title field llj
TitleBlankLLJ = dfllj['Title'].isna().sum()
print(f"Number of rows missing the '{'Title'}' field in '{'dfllj'}': {TitleBlankLLJ}")

# %%
#determing number of observations missing author and title field llj
AuthorTitleBlankLLJ = dfllj[dfllj['Author'].isna() & dfllj['Title'].isna()].shape[0]
print(f"Number of rows missing the '{'Author'}' field and '{'Title'}' field in '{'dfllj'}': {AuthorTitleBlankLLJ} ")

# %%
#getting information about spec dataframe
dfspec.describe(include='all')

# %%
#determining number of observations missing author field spec
AuthorBlankSpec = dfspec['Author'].isna().sum()
print(f"Number of rows missing the '{'Author'}' field in '{'dfspec'}': {AuthorBlankSpec}")

# %%
#determining number of observations missing title field spec
TitleBlankSpec = dfspec['Title'].isna().sum()
print(f"Number of rows missing the '{'Title'}' field in '{'dfspec'}': {TitleBlankSpec}")

# %%
#determining number of observations missing author and title field spec 
AuthorTitleBlankSpec = dfspec[dfspec['Author'].isna() & dfspec['Title'].isna()].shape[0]
print(f"Number of rows missing the '{'Author'}' field and '{'Title'}' field in '{'dfllj'}': {AuthorTitleBlankSpec} ")

# %%
#total observations across both data frames missing author and title fields 
TotalMissing = AuthorTitleBlankLLJ + AuthorTitleBlankSpec
print(f"Number of rows across both dataframes missing '{'Author'}' and '{'Title'}' fields: {TotalMissing}")

# %%
#conducting random sampling of data frames to determine if any useful data entries would be lost if some were excluded due to blank fields
rsmissingAT_dfllj = dfllj[dfllj['Title'].isna() & dfllj ['Author'].isna()].sample(n=188)
print(rsmissingAT_dfllj)

# %%
rsmissingAT_dfllj.info()

# %%
rsmissingAT_type_counts_dfllj = rsmissingAT_dfllj['Type '].value_counts()
print(rsmissingAT_type_counts_dfllj)

# %%
rsmissingAT_dfspec = dfspec[dfspec['Title'].isna() & dfllj ['Author'].isna()].sample(n=42)
print(rsmissingAT_dfspec)

# %%
rsmissingAT_dfspec.info()

# %%
rsmissingAT_type_counts_dfspec = rsmissingAT_dfspec['Type '].value_counts()
# ? what is the right variable name
print(rsmissingAT_type_counts_dfspec)

# %%
#trying to create new dataframe where all rows that were missing Author field in lljcsv are dropped
dflljauthor = dfllj.dropna(subset=['Author'])
dflljauthor.info()

# %%
#trying to create new dataframe where all rows that were missing Author field in speccsv are dropped 
dfspecauthor = dfspec.dropna(subset=['Author'])
dfspecauthor.info()

# %%
#trying to turn to authors to list for explode command, but getting warning: A value is trying to be set on a copy of a slice from a DataFrame.
dflljauthor['Author'] = dflljauthor['Author'].str.split(',')

# %%
dflljauthor = dflljauthor.explode('Author')
dflljauthor.info()

# %%
#trying to turn to authors to list for explode command, but getting warning: A value is trying to be set on a copy of a slice from a DataFrame.
dfspecauthor['Author'] = dfspecauthor['Author'].str.split(',')

# %%
dfspecauthor = dfspecauthor.explode('Author')
dfspecauthor.info()


