# kth-aml-project
Implementation of the "Text Classification using String Kernels" paper written by Lodhi, Saunders, Shawe-Taylor, Cristianini, Watkins.

### Data
Files in the ```data``` directory:

* ```train_data``` and ```test_data``` - original Reuters dataset split (Modified Apte) and (Pickled)
* ```train_data_clean``` and ```test_data_clean``` - preprocessed and cleaned dataset (Pickled)
* ```train_data_small``` and ```test_data_small``` - trimmed dataset prepared for experiments (Pickled)
* ```precomp_kernels ``` - directory with precomputed SSK gram matrices
* ```approx``` - directory with precomputed approximated-SSK files

### Setup

Before using SSK kernel compile Cython code using:
```
python setup.py build_ext --inplace
```
