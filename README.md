# Kaggle: AirBnB New User Bookings
## Problem Statement
> "New users on Airbnb can book a place to stay in 34,000+ cities across 190+ countries. By accurately predicting where a new user will book their first travel experience, Airbnb can share more personalized content with their community, decrease the average time to first booking, and better forecast demand.
>
> **In this recruiting competition, Airbnb challenges you to predict in which country a new user will make his or her first booking.**" [Kaggle link](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/overview)

## [Data](View_data.ipynb)
> The original training dataset contains 213451 rows, one for each unique AirBnB user and 16 features. These features include things like age, gender, booking dates, operating system and internet browser used to access the website, etc. For a full list of features visit [this link](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data). The training dataset contains 62096 rows and all of the same features as the training dataset, less destination_country. The training and test sets are split by dates. In the test set, we predict all the new users with first activities after 7/1/2014.
>
> Auxillary datasets include age_gender_bkts.csv, which features user age data binned into 5 year buckets and broken down by country, gender, and population size; countries.csv includes additional information about each of the destination countries; sessions.csv includes information about specific actions taken on the AirBnb website, indexed by user_id. In the sessions dataset, the data only dates back to 1/1/2014, while the users dataset dates back to 2010.

## [Data Preprocessing](Data_cleaning.ipynb)
> * Date features were converted to datetime objects, then converted to ordinal values. 
> * Rows were the first booking occured before account creation (29 observations) were dropped
> * Given the number of observations without a first booking date, first booking date was then converted to boolean
> * The age feature contained many misentered values. Entries that contained birth year instead of age, were replaced with the proper age value. Ages from below 16 and above 100 were dropped from the training set. Missing values in the training set were then imputed with the mean age of remaining observations. Missing values and values below 16 and above 100 in the testing set were also imputed with the mean of the corrected ages in the training set.
> * A feature for the time spent on the AirBnb website was added to the testing and training datasets by totalling all individual website action durations for each user, as found in the sessions dataset. Users without session information were imputed with the mean duration.
> * A feature for first language levenshtein distance from English was added by pulling information from the countries dataset. Users with a first language that wasn't found in the countries dataset were imputed with zeros (English)
> * The remaining features with categorical data were one-hot-encoded.
>
> After cleaning the training dataset contained 211041 rows and 152 columns. The testing datset contained 62096 rows and 152 columns (an empty destination_country column was added to the testing dataset when one-hot-encoding)

## [Modeling](Modeling.ipynb)
#### Splitting Data
Using train_test_split from sklearn's model_selection, we split our data into a training set and a testing set. We used all of the columns from the train data as our X features, sans the 'id' column and the target column, which was 'country_destination'.
#### Scaling
After splitting our data, we used Standard Scalar from sklearn's preprocessing library to fit and transform our X data, and transform our y data. For the MultinomialNB model, because the model cannot accept negative X values, we transformed our data using sklearn's Normalizer.
#### Initial Fit
For our initial models, we fit Logistic Regression, Random Forest, SVC, and MultinomialNB models with their default hyperparameters. 
Our accuracy scores on the train and tests splits were as follows:

| Model               | Train Accuracy | Test Accuracy |
|---------------------|----------------|---------------|
| Logistic Regression | 0.8762         | 0.8761        |
| Multinomial NB      | 0.5851         | 0.5851        |
| SVC                 | 0.8758         | 0.8774        |
| Random Forest       | 0.9963         | 0.8547        |

After reviewing the scores from our models, we decided that the Logistic Regression and Support Vector Classifier models were best suited for this type of classification problem, and moved forward with Gridsearching over different hyperparameters.

### GridSearch
**For the Logistic Regression Model, the following hyperparameters were tested:**

| Parameter | Values               |
|-----------|----------------------|
| C         | 0, 0.01, 1   |
| Solver    | 'lbfgs', 'liblinear' |
| Penalty   | 'l1', 'l2'           |

A random state of 42 was used in each fit, a 3 fold cross validation, as well as 5000 maximum iterations, to avoid convergence warnings.

**For the Support Vector Classifier, the following hyperparameters were tested:**

| Parameter | Values    |
|-----------|-----------|
| C         | 0.01, 0.1 |

A random state of 42 was used in each fit, as well as a 3 fold cross validation.
