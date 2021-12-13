from aprec.datasets.booking import download_booking_train, download_booking_test, get_booking_dataset 
from aprec.utils.os_utils import file_md5
import json

from unittest import TestCase
import unittest

class TestBookingDatset(TestCase):
    def test_booking_download(self):
        #download train file
        result_file = download_booking_train()
        booking_file_md5 = file_md5(result_file)
        self.assertEqual(booking_file_md5, "4f343b12d76b28ec0f1899e4083a72a8")
 
        #download train file
        result_file = download_booking_test()
        booking_file_md5 = file_md5(result_file)
        self.assertEqual(booking_file_md5, "2d068bea795cc4b798422ad1d80bd0c4")

    def test_booking_dataset(self):
        with open("booking_dataset_reference_actions.json") as input:
            reference_actions = json.load(input)
        actions_dataset = get_booking_dataset(max_actions_per_file=2, unix_timestamps=True)[0]
        actions = [json.loads(action.to_json()) for action in actions_dataset]
        self.assertEqual(actions, reference_actions)



if __name__ == "__main__":
    unittest.main()
        
