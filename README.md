Improved Octo Waddle
--------------------

An implementation of the balanced parentheses tree structure as described by
Cordova and Navarro (http://www.dcc.uchile.cl/~gnavarro/ps/tcs16.2.pdf).

Install notes
-------------

Installation is a two step procedure right now due to the chicken and egg
problem of requiring numpy and cython for setup.py to execute. The package is
named iow in pypi as "bp" was taken at time of registration.

```
conda install numpy cython
pip install iow
```
