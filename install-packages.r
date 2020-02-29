#==============================================================================
# 00-install-packages.R
# Purpose: install R packages and provide information about how to create
# OAuth token to query Twitter's API
# Author: Pablo Barbera
#==============================================================================

#### INSTALLING R PACKAGES FROM CRAN ####

doInstall <- TRUE  # Change to FALSE if you don't want packages installed.

toInstall <- c(
    "ggplot2", "scales", "gridExtra", "ggthemes", "RColorBrewer", ## gplot2 and extensions
    "streamR", ## library to capture and parse Tweets
    "Matrix", ## efficient storage of sparse matrices
    "reshape2", ## reshape data frames
    "devtools", ## necessary to install smappR package
    "twitteR" ## necessary to collect twitter data from REST API
    )

if(doInstall){
    install.packages(toInstall, repos = "http://cran.r-project.org")
}

#### INSTALLING R PACKAGES FROM GITHUB ####
library(devtools)
install_github("SMAPPNYU/smappR")


### REGISTERING OAUTH TOKEN ###

## Step 1: go to dev.twitter.com and sign in
## Step 2: click on your username (top-right of screen) and then on "My applications"
## Step 3: click on "Create New App"
## Step 4: fill name, description, and website (it can be anything, even google.com).
##         (Leave callback ULR empty)
## Step 5: Agree to user conditions and enter captcha.
## Step 6: copy consumer key and consumer secret and paste below

library(ROAuth)
requestURL <- "https://api.twitter.com/oauth/request_token"
accessURL <- "https://api.twitter.com/oauth/access_token"
authURL <- "https://api.twitter.com/oauth/authorize"
consumerKey <- "XXXXXXXXXXXX"
consumerSecret <- "YYYYYYYYYYYYYYYYYYY"
my_oauth <- OAuthFactory$new(consumerKey=consumerKey,
  consumerSecret=consumerSecret, requestURL=requestURL,
  accessURL=accessURL, authURL=authURL)

## run this line and go to the URL that appears on screen
my_oauth$handshake(cainfo = system.file("CurlSSL", "cacert.pem", package = "RCurl"))

## Setting working folder
## From windows machine usually this works
# setwd("H:\\credentials\\twitter")
## From Mac computer, something like...
setwd("~/credentials/twitter/")

## now you can save oauth token for use in future sessions with twitteR or streamR
save(my_oauth, file="my_oauth")