ETS Asset Management Factory Challenge 2017
===========================================

This code was written by Paul Duan (<email@paulduan.com>) and Benjamin Solecki (<bensolucky@gmail.com>) and tuned to
this challenge by Fran Lozano (<fjlozanos@gmail.com>). See [reference repository](https://github.com/pyduan/amazonaccess)
It provides my solution to the ETS Asset Management Factory Challenge.

Usage:
---------------
    [python] classifier.py [-h] [-d] [-i ITER] [-f OUTPUTFILE] [-g] [-m] [-n] [-s] [-v] [-w]

    Parameters for the script.

    optional arguments:
      -h, --help            show this help message and exit
      -d, --diagnostics     Compute diagnostics.
      -i ITER, --iter ITER  Number of iterations for averaging.
      -f OUTPUTFILE, --outputfile OUTPUTFILE
                            Name of the file where predictions are saved.
      -g, --grid-search     Use grid search to find best parameters.
      -m, --model-selection
                            Use model selection.
      -n, --no-cache        Use cache.
      -s, --stack           Use stacking.
      -v, --verbose         Show computation steps.
      -w, --fwls            Use metafeatures.


To directly generate predictions on the test set without computing CV
metrics, simply run:  

    python classifier.py -i0 -f[output_filename]

License:
---------------
This content is released under the [MIT Licence](http://opensource.org/licenses/MIT).
