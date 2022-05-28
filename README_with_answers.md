# data-science-challenge

Welcome! The following challenge is to evaluate your strengths as a data scientist. The expectation is not that you would finish all of it, but represents all of the sorts of problems you might expect working on a typical problem. You should limit the exercise to no more than a few hours. When you're finished put your code in a VCS of your choice and share the link!

## Problem Statement

We are creating a machine learning algorithm that will predict the price of Bitcoin in USD in the next second based on the price from the last 60 seconds. The variable `price_high` is the target variable we wish to predict, our *Y*. The rest of the data can be used for your feature set, *X*. I have created a lookback function to expand the feature set and a min-max scaler to standardize the features. You can use this as your feature set, or modify it as you wish. The dataset consists of the first 10,000 seconds of Bitcoin data a single day, January 1, 2020 (until about 8:00 am). The data can be found in `bitcoin.csv`. We have a model in place, trained in `bitcoin-predictor.ipynb`. I started by using a deep neural network, because everyone keeps talking about how cool they are. Problem is, the model is not very good. Actually the model is completely awful and predicts `0` for everything. So, a lot of room for improvement!

The main challenge is this: can you make a model that actually works? You can use a neural network in tensorflow as I've done, or you can scrap it altogether and use regression, random forest, svm, anything you like. No model or package is off the table. Feel free to modify the jupyter notebook directly or put it in whatever workflow suits you best.

The main idea is to get a decent working model up and running quickly, but some other considerations you should think about are listed below. You don't have to answer all of the questions, but pick at least one or two and try to answer them (you can add your answers in this readme if you'd like).

* Does this data set even make sense? What are the limitations of this data set?

The data does make sense.  I had to infer some meaning from the column labels, but I'm pretty sure I understood it.

The data looks at only a short window of bitcoin price.  Given how volatile the price of bitcoin is, I would be concerned about new data in vastly different price regimes (half/double the training prices or more) as well as data during periods of extreme volatility.

* Is the lookback window of 60 seconds helpful? What are its limitations? What other features would you want to see in this data set?

The lookback window is necessary to the problem statement.  A major limitation I see is data leakage.  Data from the test set indirectly leaks into the train set in different features.  For instance, two consecutive seconds of data will have 59 identical feature values (shifted by a column).

I don't know enough about how much more granular this data can go, but it may be important to know the specific prices of each trade during the one second intervals (and the exact time they happened).  I think that may be overkill though.

* If you stuck with the neural network, what did you change to make it better? Did you change the architecture, did you change the optimizer? The learning rate? The activation function(s)? Why was the model stuck at `0` with an incredibly high root mean squared error?

I don't know much about using different architectures or optimizers within a neural network.  I do know enough about learning rate and activation function to suspect at least one of those would be the culprit if the predictions are all zero (to me that indicates a model that isn't learning at all, rather than a model that isn't optimal).  I toyed with the learning rate a bit to no avail and then used google to investigate activation functions.  I had never heard of LeakyReLU before, but it seems to have done the trick.

I won't try to explain why the outputs were all zero beyond the observation that the model had trouble learning with just ReLU.  I believe it's related to the horizontal regime of ReLU and the vanishing gradient problem.

* If you used a different model, why'd you choose this model? What about it made it work for this problem? Is this model complex and if so, is the complexity necessary? Is it intuitive enough to explain it to a lay-person? What was your optimizing metric? What were the hyperparameters and why'd you choose them?

I also used a Random Forest for comparison.  I tried to choose the simplest model possible that wouldn't have an obvious limitation (with linear regression we'd have to discard every feature except the prices since the coefficients for trade volume should be positive or negative depending on how the price is shifting).  That limitation on linear regression points to the need for a tree of some sort.

I think a Random Forest is easier to explain than a LSTM model.  I did not spend any additional thought on hyperparameters.

* Did you include any regularization strategies in your model? If so, why'd you choose the one you did?

I did not alter regularization beyond the included early-stopping for LTSM.

* Did you include visualizations? (everyone loves a good graphic)

Of course!

* How do we know the model is good? How understandable are the diagnostics? How will we know how good the model is predicting in production?

The trained models (LTSM and RF) beats two naive predictors I used in comparison.  One just predicts that the next price will be the same as the previous price.  The other averages across the past 60 seconds of prices.  If we couldn't beat those, we'd be chasing our own tail.

A dashboard would be the ideal way to monitor how the model is predicting in production.

* If we see data for more than a single day's worth of prices, how do expect the model to perform? Will it generalize well to new data? Will retraining with this new data be an issue for this model?

I expect issues with new data because a single day hardly encompasses the story of bitcoin prices.  It would be interesting to see how the model performs on data for extremely volatile time periods.

* **What question would you ask of the data, or add to this analysis that I haven't thought of?**

I will take this time to once again emphasize my concern about data leakage arising from the lookback window.


## Note on Dependencies

Feel free to install dependencies however you wish (docker, pyenv, conda, etc.). This model was trained in a conda virtual environment, and you can find all of the details in the `spec-file.txt` (the output of `conda list`) file or the explicit packages in `spec-file-explicit.txt` (the output of `conda list --explicit`). You may upgrade/downgrade dependencies as you see fit, as long as it doesn't effect the solution.
