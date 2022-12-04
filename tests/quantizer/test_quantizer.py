import unittest

from aprec.datasets.datasets_register import DatasetsRegister
from aprec.datasets.movielens20m import get_movies_catalog
from aprec.quantizer.quantizer import Quantizer


class TestQuantizer(unittest.TestCase):
    
    def test_quantizer(self):
        actions = DatasetsRegister()["ml-20m_warm5_fraction_0.01"]()
        quantizer = Quantizer()
        quantizer.fit(actions)
        catalog = get_movies_catalog()
        self.print_similar_movies(quantizer, catalog, '1') #Toy Story
        self.print_similar_movies(quantizer, catalog, '1240') #Terminator
        self.print_similar_movies(quantizer, catalog, '69122') #Hangover

    def print_similar_movies(self, quantizer, catalog, movie_id):
        most_similar = quantizer.sim(movie_id) 
        for rec in most_similar:
            print(catalog.get_item(rec[0]), "\t", rec[1])
        print('====')



if __name__ == "__main__":
    unittest.main()

