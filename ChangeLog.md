BP ChangeLog
============

BP 0.1.1-dev
------------

* Add Python 3.13, 3.14, and newly released 3.15 to CI for Linux and macOS.
* Declare isolated build dependencies in pyproject.toml so setup.py can import
  NumPy, Cython, and Versioneer during modern pip builds.
* Replace removed pkg_resources usage in test-data helpers with importlib.resources.

BP 0.1.1 (released on August 26, 2016)
--------------------------------------

* mucking around to make it actually pip installable...

BP 0.1 (released on August 26, 2016)
------------------------------------

* initial release. Not complete, but reasonably fast and memory efficient. This
    release was made as the object is minimally sufficient for its original
    intended use: optimization of UniFrac.
