# kth-aml-project
Implementation of the "Text Classification using String Kernels" paper written by Lodhi, Saunders, Shawe-Taylor, Cristianini, Watkins.

### Data
The ```data``` directory has two data files ```train_data``` and ```test_data```.

These files are in the Pickle format and hold the dataset split as suggested in the paper (Modified Apte spit).

### Setup

To compile SSK, run

```
python setup.py build_ext --inplace
```
