## Item-item CF
#%%
# Libraries
import pandas as pd
import numpy as np

#%%
# read in data
ratings_df = pd.read_excel(
    'Assignment 5.xlsx',
     sheet_name='Ratings',
     headers=True
)
ratings_df.head()
#%%
# cleanup
# make the users variable names into strings
ratings_df['User'] = ratings_df['User'].astype(str)
# set users as the row index
ratings_df = ratings_df.set_index('User')
# get a list of moves and users, can be useful for setting
# columns and indices later
movies = ratings_df.columns
users = ratings_df.index
ratings_df.head()

#%%
# We need to find which weights correspond to actual ratings
# Create a binary matrix where 1 = rated, 0 = unrated
rated_df = ratings_df.copy()
rated_df[rated_df > 0] = 1
rated_df.head()

#%%
# Do we have numbers?
assert all(ratings_df.dtypes == float), "all the columns are floats"

#%%
# user average ratings
user_avgs = ratings_df.mean(1)

# normalised ratings, ratings minus user average rating
ratings_norm_df = ratings_df - user_avgs[:,None]
ratings_norm_df.head()

#%%
# L2 norms
l2_norms = np.sqrt((ratings_df*ratings_df).sum(0))
l2_norms_norm = np.sqrt((ratings_norm_df*ratings_norm_df).sum(0))

#%%
# get item-item correlations
dotp = (ratings_df.fillna(0) / l2_norms)
dotp = dotp.transpose() @ dotp
# get item-item normalised correlations
dotp_norm =  (ratings_norm_df.fillna(0) / l2_norms_norm)
dotp_norm = dotp_norm.transpose() @ dotp_norm

#%% 
# clamp to zero
weights = dotp.clip(0, inplace=False).fillna(0)
#np.fill_diagonal(weights.values, 0)
weights_norm = dotp_norm.clip(0, inplace=False).fillna(0)
#np.fill_diagonal(weights_norm.values, 0)

#%%
weights.head()

#%%
weights_norm.head()

#%%
# check that the columns and indices are in order
assert all(ratings_norm_df.columns == ratings_df.columns), "ratings columns are not equal"
assert all(ratings_norm_df.columns == rated_df.columns), "rated_df columns are not equal"
assert all(ratings_norm_df.columns == weights.columns), "weights columns are not equal"
assert all(weights_norm.columns == weights.columns), "weights_norms columns are not equal"

assert all(ratings_norm_df.index == ratings_df.index), "ratings indices are not equal"
assert all(ratings_norm_df.index == rated_df.index), "rated_df indices are not equal"
# weights have the same columns as indices
assert all(weights_norm.index == weights.index), "weights indices are not equal"


#%%
# item-item collab filtering results
ii_cf_df = (ratings_df.fillna(0)) @ weights / (rated_df.fillna(0) @ weights)
ii_cf_df.min(), ii_cf_df.max()

#%%
# item-item collab filtering normalised results
ii_cf_norm_df = (
    (ratings_norm_df.fillna(0) @ weights_norm)
     / (rated_df.fillna(0) @ weights_norm)
     + user_avgs[:,None]
)
ii_cf_norm_df.min(), ii_cf_norm_df.max()

#%%
# control answers 1
print("5 most similar movies to toy story.")
print("raw")
# first one will be self correlation of 1, can ignore
print(weights["1: Toy Story (1995)"].sort_values(ascending=False)[1:6])
# normalised
print()
print("normalised")
print(weights_norm["1: Toy Story (1995)"].sort_values(ascending=False)[1:6])

#%%
# control answers 2
print("Top 5 movies for user 5277")
print("raw")
print(ii_cf_df.loc["5277",:].sort_values(ascending=False)[:5])
print("")
print("normalised")
print(ii_cf_norm_df.loc["5277",:].sort_values(ascending=False)[:5])
#%%
