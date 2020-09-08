Vehicle Accident Prediction Model 

  

Overview  

The possible causes of Accidents are many but if we look closely, there are patterns where accidents are more likely to happen. We notice that accidents are more in news during stormy weather/thunderstorms. If we can identify these patterns closely, a lot of these events could be prevented. This was the objective of the project. I wanted to see if Machine Learning could hep in identifying the underlying causes. 

  

Data 

The data was taken from Kaggle. It contained US Accident data over a period of 5 years. Due to huge size, I could not use the whole dataset for training purpose. I chose Texas as the choice of State. But while testing, I used the complete data set. This will be discussed further under Cross-Validation. 

  

Data Analysis/Feature Extraction 

I found good insights from data visualisations which helped me in choosing the relevant features. 

For example, there are significantly higher number of accidents in the weekdays compared to weekends. Also, the accidents are more in August until december due to colder weather and correspondingly adverse weather conditions. The above data also correlated well with the population data of cities in Florida. 

The weather conditions were given in a long text format so I used NLTK Library to extract the "Weather Keyword". Also, the latitude and longitude features are relevant but they have lot of variability. Instead I used DBScan clustering to cluster all accident points into divisions and used ony that cluster as a feature.  

Since the data only had accident points, I manually created non accident points by varying few features in the existing dataset. This gave a well balanced set with no overfitting. 

  

Model Selection/Error Metric 

I used Random Forest and XGB as the base models. They both had similar logloss errors but XGB was much faster in training. Performed RandomisedSearchCV on both monitoring the Log loss error and AUC Score. Here XGB performed much better and the model size was comparatively smaller. The confusion matrix showed the F1 score as 0.88 for No Accident Scenario.  

  

Model Deployment 

The model was deployed on a local server using Flask. The webpages were created and linked to Flask functions. The final webpage takes in the route lat long details and certain weather conditions to predict if that route is safe or not.  

  

Conclusion and Further Steps 

The goal of creating an accurate pattern recognition for accidents was successful. The accuracy achieved was very good (0.92). I further used the model to identify the accident prone clusters(Hotspots) and figure out the features which make each of them unsafe. This could help organisations to take note of what changes could be done to prevent these. Also, we could identify accident hotspots with similar features using clustering/Pearson's method, so that developments in one locality could be used to help other localities. 

Another thing, I would like to try is to create an end to end app, where a user enters the starting and ending point and the time frame in which he wants to travel. The app could determine the best time to take the journey based on weather, accident-predictor model, night-time considerations etc. 
