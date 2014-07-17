# CDIPS Workshop 2014: Food vs. Skin

### Team Members

* Kaylan Burleigh
* Maurizio Pellegrino
* C.Al Rive
* W. Todd Doughty

### Project Mentor

* Dr. Sudeep Das (OpenTable)

### Project Description

As a Data Scientist at OpenTable, my computer screen often fills up with images of scrumptious food items (effectively keeping my metabolic rate on a high gear!). Many of these photographs are professionally or semi-professionally taken by food photographers or enthusiasts (aka FoodSpotters). Annoyingly, a lot of photos, especially those from social media channels, come  with portraits of eaters posing with the eaten! Of course, one could use face detection algorithms, and other sophisticated techniques to weed out these photos. These techniques often require a lot of overhead and dependency on external  libraries. Also, there may not even be a face in the photo to detect, but a hand or some portion of the torso might be showing. The problem is to find a smart set of algorithms that can detect skin  pixels and/or faces, and effectively detect and weed out photos that have people or parts of people in them. 

You can look at my blog post as a starting point to get a sense of the problem: 

http://datamusing.info/blog/2014/07/06/detecting-people-in-photographs-using-skin-tone/

This project will involve unsupervised as well as supervised learning, feature generation, possibly some Fourier space voodoo, signal processing and a chance to work with tons of real data, of course!

*From email from Sudeep Das*

### Project Deliverables

An interface where a user can upload a list of picture files and get a classification of those containing/not containing skin.

This entails the following:

1 Develop data prep algorithm
  * Check data quality
2 Modeling approaches
  * Using machine learning (supervised and/or unsupervised) to classify the pictures
3 Optimal algorithm implementation and evaluation
  * Evaluating the algorithm on an independent test set