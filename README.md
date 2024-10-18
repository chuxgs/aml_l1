Activity 1. Practice with GLMs

1. Logistic regression

2. The whole documentation about scikit logistic regression's implementation in Python can be found [here](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html).
In general there is only one required parameter which is data as lots of other "required" parameters are set to default values in case user is not specifying them, but
obviously we have full control of them (for instance solver type, regularization penalty, class(es) weights, etc.)

3. The data set was taken from [here](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction). 
The target variable is in the column "left" where 1 means that an employee left the company and 0 if he/she did not. Train/test split is decided to be 70/30.

4. Since the task required use to use SGD solver (sag or saga in the case of python's implementation), the only regularization that we can include is two norm (l2).
If we had tried more solvers we could have also tried one norm(l1). We also tried l1 norm for the saga solver as it is possible, but
l2 results in the better model on both train and test data, so we decide to drop l1 norm to not output warnings messages for the sag solver. We cannot calculate the AIC value of the best model directly using
scikit learn library, but we can still do it as we have access to the data and the best model's parameters (their number in particular).

5. We dropped duplicate rows, we did one hot encoding of the categorical variable (salary was just changed to the numbers as it has meaningful order),
we balanced the training data using SMOTE (oversampling method, maybe undersampling or using non-default class weights is a better idea for the performance of the model),
and lastly we scaled the data.

6. The Pipeline and GridSearchCV from scikit learn allows you to get the best model (with the best hyperparameters if specified) based on the given scoring metric.
In our code we used accuracy as the scoring metric and cross validation with k=5

7. The results definitely suggest that the improvement is possible. Most likely this can be done by discarding irrelevant variables instead of using all 17 (we haven't really thought what's the best approach to do that), and by taking more control of other model's parameters.

