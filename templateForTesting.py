# Template for using the 'unittest.py' module for testing python code
# documentation can be found here: https://docs.python.org/3/library/unittest.html
# Note that there are many other functions available, see the docs

import unittest

# Replace 'MyClass' with the name of your class
# and write your constructor and methods inside this class
class MyClass: # 'MyClass' should be your class name

    def __init__(self, name):
        # class constructor, takes one string input parameter
        self.name = name
    
    def myFunc(self, param):
        # An example method, replace 'myFunc' with your method name
        # and put the calculations in here
        return 1 + param

# Here is the class to perform the tests
# Rename 'TestMyClassMethods' to something of your own choice
# 'unittest.TestCase' should stay the same
class TestMyClassMethods(unittest.TestCase):

    def setUp(self):
        # This code will be run before anything else
        # and it creates an instance of your class
        self.myObject = MyClass('objectName') # replace with your class name and object name

    # Here you put any number of testing functions that you can name yourself
    def test1(self): # Rename 'test1' to something more descriptive
        # The next two lines verify that the two parameters are equal
        # so use this with your class methods and their expected results
        self.assertEquals(self.myObject.name, 'objectName')
        self.assertEquals(self.myObject.myFunc(1), 2)

# These lines mean that if you run this file directly, it will run the unit tests
# but if you use your class in another file, the tests will not be run
if __name__ == '__main__':
    unittest.main()