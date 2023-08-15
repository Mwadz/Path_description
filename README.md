# Path Description Model

## BART XML PATH DESCRIPTION GENERATION

### Business Understanding
Elevated solutions LLC  needs a machine learning model that takes in an XML path and gives out or generates a description of the path.

### Data Understanding
The data is in .txt files whereby we had one training file and an evaluating file. They are both similar in structure though the contents of the description and xml path are unique to prevent overfitting.

Data Preparation
Our train data was collected from gridml’s public library. It was not scraped but rather manually collected to maintain integrity and remove phrases such as examples, explanations of how the for example “file name” was changed to “name” and so on. 

The paths retained their special characters while the description special characters got dropped. Majority were commas. They had to be removed so the files could be read in csv(comma separated values) form. The files were loaded into a dataframe using the pandas library

The columns were then given new labels. The descriptions were labelled as “target_text” while the XML paths were labelled as “input_text”.


Modelling
I employed the use of a pre-trained bart model as the baseline model. This bart model in particular was developed by facebook. It uses both BART and transformer algorithms which are traditionally used in text summarization which essentially is what we’re trying to make our model do. Taking from the path only that which is meaningful and generating an explanation (description). 

I used the simple transformers library and  the sequencer sequence class model(Seq2seq) to import the model and the Seq2seq arguments.


The model was about 1.2GB in weight which is quite heavy. The training would take up to 5 hours without the use of the cloud GPU while it takes an hour with the resource. Finding a hack whereby you specify the use of cuda GPU within the code where we download the model helped save a lot of time and the model trained in under an hour after adding that line of code


One major challenge was training while having exhausted the Google collab GPU resource. It took too long and would disconnect from runtime once it detected inactivity for a while. This was a major time waster. The use of the cuda code line in the model bypassed that issue by directing the training towards the available GPU thus increasing the speed of learning. Amazon Web Service however should be considered for long term use.

Evaluation

I trained it using 10 epochs. The  number of epochs is an important hyperparameter for the algorithm as it specifies the number of complete passes of the entire training dataset passing through the training or learning process of the algorithm. Simply put an epoch in machine learning means one complete pass of the training dataset through the algorithm.

The model took an average of 20minutes to load without a GPU and under 1 minute to load on one. More specifically 40 seconds.  According to the graph, the optimum number of epochs is 4 for our data as it results in a low running loss. Although epoch 8 is the lowest running loss, the fact that it rose before shooting down suggests overfitting.

When the model memorises the noise and fits too closely to the training set, the model becomes “overfitted” and it is unable to generalise well to new data. If a model cannot generalise well to new data, then it will not be able to perform the prediction tasks that it was intended for.

The line graph below shows the running loss of each epoch. Epoch 1 was at a running loss of 5.1 therefore excluded from the graph


I then ended the evaluation by asking the model to describe the below file path:
/DOCUMENT/DEAL_SETS/DEAL_SET/DEALS/DEAL/ASSETS/ASSET/OWNED_PROPERTY/PROPERTY/IMPROVEMENT/CAR_STORAGES/CAR_STORAGE/AdequateIndicator

To which it predicted:
[' Indicates whether the car is adequate for normal needs.']

Gridml’s description of the exact same XML path is as follows:

/DOCUMENT/DEAL_SETS/DEAL_SET/DEALS/DEAL/ASSETS/ASSET/OWNED_PROPERTY/PROPERTY/IMPROVEMENT/CAR_STORAGES/CAR_STORAGE/AdequateIndicator; Indicates that this feature is adequate for normal needs.

This suggests that our model has a lot of potential for accuracy given enough examples, with optimum parameter tuning and computational power
Deployment
For now the prediction can be done directly on the notebook but I would suggest having a user interface in order to interact with the model seamlessly. The model is to be saved then loaded into the web application which can be created using the various languages and platforms.
Conclusion
The model performs a lot better than the one in phase 1 due to the cleanness, the increase in the amount of the data and the cuda code. It needs hyperparameter turing and the use of the number of epochs to data ratio discovered in order to improve accuracy and reduce running and train loss. The use of AWS cloud services could reduce the amount of training time significantly and provide the opportunity to train a whole lot more data. 
Recommendations
The use of the epoch-data ratio discovered (1250:4)
The performance of hyperparameter tuning and the discovery of which hyperparameters are most crucial to what we’re trying to achieve
The research and use of AWS GPU instances.
The consideration of software deployment for this model


