What is Review2
Review2 is a chain of ML-based models which sequentially evaluates a given review to detect for quality related policy violations and flag it out for further review if needed.

Our solution runs in two stages:
 - In the first stage, we use simple models and rules based filters to flag out our more simple policies such as spam, ads, and hate content.
 - In the second stage, we use some more complex models to filter for some more indistinct factors such as how relevant the review is and if it was written second-hand.

Set Up Instructions
1. Download trained model data
2. Run the python script
3. Presto!

Replication
1. Find or make  a dataset of online reviews in CSV format
2. Ensure it has text, star ratings (ratings scale from 1-5, with 1 being the worst) and location
3. Text and star ratings should be in the columns "text", "ratings" and "location" respectively
4. Train the models on the dataset
