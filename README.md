# Carrefour Hackathon 2021: Image (Product) Recognition

**Team:**

- Ashraf GHIYE
- Vanessa CHAHWAN
- Rania FERCHICHI
- Mohamed ALI JEBALI
- Karim SIALA

 
# Problem Description

The objective of this challenge is to be able to identify a product category from photos taken by users.

Products can be categorized into different categories and through many levels. For example, a bottle of Badoit can be first classified as Drinks or (or Boissons), but it has also more nested levels such as Water (i.e. level 2: Eaux) and more specifically sparkling water (i.e. level 3: Eau gazeuses) and finally a fourth level that is very specific for this mark of water.

In the context of that challenge, we are only interested in predicting level 3 (a multi-class problem with 417 different classes).

Refer to report for more explanation.


# Repo Organization


The first notebook `metadata_exploration_I` focuses on data exploration and visualization of the distribution of the features.

The second one `image_exploration_II` focuses on the images. Here we take a look at the images, the views and dump the ones we decided to keep for training, withe size that we wanted.

The final `model_and_evaluation_III` notebook and the most important one contains the model, and it's evaluation, both on validation and test set.

To keep the code as simple as possible, the python script `util.py` contains all the functions we used for cleaning, dropping, plotting etc…

