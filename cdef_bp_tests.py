import bp.tests.test_bp_cy

for n in dir(bp.tests.test_bp_cy):
    if n.startswith('test_'):
        getattr(bp.tests.test_bp_cy, n)()
