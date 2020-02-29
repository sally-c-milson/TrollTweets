#==============================================================================
# 01-collect-tweets.R
# Purpose: illustrate how tweets were collected, and provide code to re-download
# tweets using tweet IDs (provided)
# Author: Pablo Barbera
#==============================================================================

# The following code illustrates how one of our collections was created.
# However, note that tweets can only be created in real time. Given Twitter's
# Terms of Service, we cannot share the entire dataset, but instead we provide
# code that shows how each of the collections can be recreated.

library(ROAuth)
library(streamR)
keywords <- c("obama", "romney")

# this script will be run once every hour, and tweets are stored in different
# files, whose name indicate when they were created.
current.time <- format(Sys.time(), "%Y_%m_%d_%H_%M")
f <- paste0("Election_", current.time, '.json')

# loading OAuth token
load("~/credentials/twitter/my_oauth")

# open connection to Twitter's Streaming API
filterStream(file.name = f, track = keywords, timeout = 60*60, oauth = my_oauth)

#####################################################
#### REPLICATION: TABLE 1
#####################################################

# Since we cannot share the full tweets datasets, instead we provide the list
# of tweet IDs, which can be used to (a) replicate Table 1 of the paper;
# and (2) if desired, reconstruct the dataset of tweets.

# Replication of Table 1
system("wc -l tweet-collections/*")

# Data: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PDI7IN

# How to recover tweets by ID
library(smappR)

## Example: reading first 100 tweets
ids <- scan("tweet-collections/MinimumWage.txt", n=100, what="character")

## downloading statuses
getStatuses(ids=ids, filename='minimum-wage-tweets.json',
    oauth_folder = "~/credentials/twitter")

## reading tweets in R
library(streamR)
tweets <- parseTweets("minimum-wage-tweets.json")

# (total of tweets will be lower because of deleted tweets, deactivated
# accounts, etc.)
