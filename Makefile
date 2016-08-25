libbitarr.a: 
	cd BitArray && $(MAKE)
	make -C BitArray -f Makefile

build: libbitarr.a
	python setup.py build_ext --inplace

test: build
	python -c "from bp._binary_tree import test_binary_tree; test_binary_tree()"
	python cdef_bp_tests.py
	nosetests

clean:
	rm -fr bp/*.so bp/*.c
	rm -fr build
