# String Kernels
Implementation of the ["Text Classification using String Kernels"](http://www.jmlr.org/papers/volume2/lodhi02a/lodhi02a.pdf) publication by Lodhi et al. Code was written mainly in Python with some parts moved to Cython for performance gains. The final report can be found [here](final-report.pdf).

This project was carried out as part of the DD2434 "Advanced Machine Learning" course at [KTH Royal Institute of Technology](http://kth.se).

## Contributors
- F. Franzen (github: [flammi](https://github.com/flammi))
- B. Godefroy (github: [BGodefroyFR](https://github.com/BGodefroyFR))
- W. Kryściński (github: [muggin](https://github.com/muggin/))
- V. Polianskii (github: [vlpolyansky](https://github.com/vlpolyansky))


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
