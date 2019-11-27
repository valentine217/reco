#%%
# Libraries
import pandas as pd
import numpy as np

#%%
# read in data
ratings_df = pd.read_excel(
    'UUCF Assignment Spreadsheet.xls',
     sheet_name='movie-row',
     headers=True
)
ratings_df.head()
#%%
# cleanup
ratings_df = ratings_df.set_index(ratings_df.columns[0])
ratings_df.index.name = "movie"
users = ratings_df.columns
movies = ratings_df.index

# make the columns names into strings
users = [str(x) for x in users]
ratings_df.columns = users

ratings_df.head()

#%%
# Do we have numbers?
ratings_df.dtypes

#%%
# get user-user correlations
cor_df = ratings_df.corr()
# fill the diagonal with 0 (we don't care about self-correlation)
np.fill_diagonal(cor_df.values, 0)
cor_df.head()

#%%
# get top 5 neighbours (columnwise)
neighbours_df = cor_df.copy()

for col in cor_df.iteritems():
    top_5 = np.sort(col[1])[-5]
    neighbours_df.loc[
        col[1] < top_5,
        col[0]
    ] = 0

neighbours_df.head()
#%%
# check that it went columnwise
assert min(neighbours_df.apply(lambda col: col[col>0].count(), axis=0)) == 5, (
    "The minimum number of top 5 is not 5"
) 
assert max(neighbours_df.apply(lambda col: col[col>0].count(), axis=0)) == 5, (
    "The maximum number of top 5 is not 5"
)

#%%
# each column in neighbours_df cannot map to a row in ratings_df
# we have to sort the data first
neighbours_df = neighbours_df.loc[users,users]
ratings_df = ratings_df.loc[movies, users]

#%%
# Now each row of ratings can multiply with a column in neighbours
# first fillna with 0
dotprod = ratings_df.fillna(0) @ neighbours_df.fillna(0)

# to calculate the weights we have to know if the person reviewed a movie
# create a 1/0 matrix from ratings
rated_df = ratings_df.fillna(0).values
rated_df[rated_df > 0] = 1
weights = rated_df @ neighbours_df.fillna(0).values
# user-user collab filt results
uu_cf_df = dotprod / weights

uu_cf_df.head()

#%%
# answers for quiz
print(uu_cf_df.loc[:,'89'].sort_values()[-3:])
print(uu_cf_df.loc[:,'3867'].sort_values()[-3:])

#%% 
#### Normalised
avg_user_ratings_df = ratings_df.mean(0)
ratings_norm_df = ratings_df - avg_user_ratings_df

#%%
dotprodnorm = ratings_norm_df.fillna(0) @ neighbours_df.fillna(0)
uc_cf_norm_df = dotprodnorm / weights + avg_user_ratings_df

#%%
# answers for quiz
print(uc_cf_norm_df.loc[:,'89'].sort_values()[-3:])
print(uc_cf_norm_df.loc[:,'3867'].sort_values()[-3:])
