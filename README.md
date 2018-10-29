## Project 1 for Machine Learning EPFL - Higgs Boson

In this project, we implemented different kind of regression techniques to implement this classification task.


- **implementations.py:** contains all the implementations of each of the regression methods and their corresponding loss functions.
- **running_grid_search.py:** contains the logistic regression code which runs grid search on several hyper parameters and does cross cross validation.
- **run_prediction.py:** contains the regularized logistic regression code which trains the model on the final parameters and generates a `categorized` test.csv file to submit on kaggle.
- **Report.pdf:** contains the 2 page report for this project.
- **proj1_helpers.py:** contains some helper functions to load or sort the csv file, predict labels.
- **Project1_ML.ipynb:** is IPython Notebook in which we did the initial experimentation with several methods and also did some correlation related feature engineering experimentation.
- **categroy.py:** script which given the training dataset divides into different categories based on the `JET numbers` and return a categorized dataset.


To run any of the scripts such as `running_grid_search.py` or `run_prediction.py`, we should just call it in the terminal with `python <name-of-the-script>`.
