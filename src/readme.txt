INTRODUCTION
=============
This project explores NCI GDC dataset to filter out the samples of mislabeled gender.
We use two different approchees to figure out mislabling:

1. use supervised Machine Learning to train a model and to predict gender based on miRNA data
2. use unsupervised to do clustering within gender group, then split into minority group and majority group.

We take the smaples that mismatched with predited gender and belong to miniroty group as mislabed samples.

HWO TO RUN
============
1. git clone https://github.com/uscwy/ee542lab10
2. Copy source code to src directory in ee542lab10
3. Run Jupyter notebook

SOURCE CODES
=============
  mislabel.ipynb        read miRNA_matrix.csv as data source, try to find mislabed samples
  gen_miRNA_matrix.py   generate miRNA data with gender and primary site
  preprocess.ipynb      convert Gene Expression Quantification data to dataframe
  lab9.ipynb            for lab9  
  
DATA FILES
============
  miRNA_matrix.csv      miRNA data file with gender and primary site
  
