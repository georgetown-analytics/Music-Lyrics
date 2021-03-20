"""
Foo package tests.
"""

import pytest

class TestInitialization(object):
	def test_sanity(self):
		'''
		Test that tests work by confirming 2**3 = 8
		'''
		assert 2**3==8, "The world went wrong!"

	def test_import(self):
		'''
		Assert that the conduit package can be imported.
		'''
		try:
			import foo
		except ImportError:
			pytest.fail("Coul;d not import from the foo library!")
