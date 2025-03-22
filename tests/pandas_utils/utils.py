import unittest
from unittest import TestCase

from pandas import DataFrame

from mitoolspro.pandas_utils.utils import remove_dataframe_duplicates


class TestRemoveDataframeDuplicates(TestCase):
    def setUp(self):
        self.df1 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        self.df2 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        self.df3 = DataFrame({"A": [7, 8, 9], "B": [10, 11, 12]})

    def test_remove_dataframe_duplicates(self):
        dfs = [self.df1, self.df2, self.df3]
        unique_dfs = remove_dataframe_duplicates(dfs)
        self.assertEqual(len(unique_dfs), 2)
        self.assertTrue(unique_dfs[0].equals(self.df1))
        self.assertTrue(unique_dfs[1].equals(self.df3))

    def test_remove_dataframe_duplicates_no_duplicates(self):
        dfs = [self.df1, self.df3]
        unique_dfs = remove_dataframe_duplicates(dfs)
        self.assertEqual(len(unique_dfs), 2)
        self.assertTrue(unique_dfs[0].equals(self.df1))
        self.assertTrue(unique_dfs[1].equals(self.df3))

    def test_remove_dataframe_duplicates_all_duplicates(self):
        dfs = [self.df1, self.df1, self.df1]
        unique_dfs = remove_dataframe_duplicates(dfs)
        self.assertEqual(len(unique_dfs), 1)
        self.assertTrue(unique_dfs[0].equals(self.df1))


if __name__ == "__main__":
    unittest.main()
