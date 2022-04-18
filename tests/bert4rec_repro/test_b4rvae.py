from unittest import TestCase
import unittest
import b4rvae.main as main

class TestB4RVAE(TestCase):
    def test_b4rvae(self):
        main.train()

if __name__ == "__main__":
    unittest.main()