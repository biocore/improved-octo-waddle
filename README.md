Improved Octo Waddle
--------------------

An implementation of the balanced parentheses tree structure as described by
[Cordova and Navarro](http://www.dcc.uchile.cl/~gnavarro/ps/tcs16.2.pdf).

Install notes
-------------

Installation is a two step procedure right now due to the chicken and egg
problem of requiring numpy and cython for setup.py to execute. The package is
named iow in pypi as "bp" was taken at time of registration.

```
$ conda create --name bp python=3.8
$ conda activate bp
$ conda install numpy cython
$ pip install iow
```

Developer notes
---------------

If pulling the source, please note that we use a submodule and Github does not
by default bring it down. After a clone, please run:

```
$ git submodule update --init --recursive
```

Fragment insertion
------------------

BP supports the [jplace format](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0031009). Fragments can be inserted using either fully-resolved or multifurcation mode to resolve multiple placements to the same edge. In fully resolved, the edge placed against is broken N times where N is the number of fragments on the edge. In multifurcation, a new node is constructed as the average of the distal length for the N fragments, and a separate multifurcation node is added which encompasses the placed fragments.

Insertions can be handled by the command line following install:

```
$ bp placement --help
Usage: bp placement [OPTIONS]

Options:
  --placements PATH               jplace formatted data  [required]
  --output PATH                   Where to write the resulting newick
                                  [required]

  --method [fully-resolved|multifurcating]
                                  Whether to fully resolve or multifurcate
                                  [required]

  --help                          Show this message and exit.
```

Note that the multifurcating support relies on GPL code derived from the Genesis project. That code and LICENSE can be found under `bp/GPL`.
