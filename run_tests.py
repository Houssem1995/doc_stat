import unittest
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Discover and run tests
loader = unittest.TestLoader()
start_dir = os.path.join(os.path.dirname(__file__), 'tests')
suite = loader.discover(start_dir, pattern='test_*.py')

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

# Exit with non-zero code if tests failed
sys.exit(not result.wasSuccessful()) 