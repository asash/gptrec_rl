from aprec.datasets.booking import download_booking_train, download_booking_test, get_booking_dataset 
from aprec.utils.os_utils import file_md5

from unittest import TestCase
import unittest

class TestBookingDatset(TestCase):
    def test_booking_download(self):
        #download train file
        result_file = download_booking_train()
        print(result_file)
        booking_file_md5 = file_md5(result_file)
        self.assertEqual(booking_file_md5, "4f343b12d76b28ec0f1899e4083a72a8")
 
        #download train file
        result_file = download_booking_test()
        print(result_file)
        booking_file_md5 = file_md5(result_file)
        self.assertEqual(booking_file_md5, "2d068bea795cc4b798422ad1d80bd0c4")

    def test_booking_dataset(self):
        reference_actions = "[Action(uid=1000027_1, item=8183, ts=2016-08-13 00:00:00, data={'user_id': '1000027', 'device_class': 'desktop', 'affiliate_id': '7168', 'hotel_country': 'Gondal', 'booker_country': 'Elbonia', 'checkin_date': datetime.datetime(2016, 8, 13, 0, 0), 'checkout_date': datetime.datetime(2016, 8, 14, 0, 0), 'is_control': False}), Action(uid=1000027_1, item=15626, ts=2016-08-14 00:00:00, data={'user_id': '1000027', 'device_class': 'desktop', 'affiliate_id': '7168', 'hotel_country': 'Gondal', 'booker_country': 'Elbonia', 'checkin_date': datetime.datetime(2016, 8, 14, 0, 0), 'checkout_date': datetime.datetime(2016, 8, 16, 0, 0), 'is_control': False}), Action(uid=1000066_2, item=56430, ts=2016-07-21 00:00:00, data={'user_id': '1000066', 'device_class': 'desktop', 'affiliate_id': '9924', 'hotel_country': 'Urkesh', 'booker_country': 'Gondal', 'checkin_date': datetime.datetime(2016, 7, 21, 0, 0), 'checkout_date': datetime.datetime(2016, 7, 23, 0, 0), 'is_control': True}), Action(uid=1000066_2, item=41971, ts=2016-07-23 00:00:00, data={'user_id': '1000066', 'device_class': 'desktop', 'affiliate_id': '9924', 'hotel_country': 'Urkesh', 'booker_country': 'Gondal', 'checkin_date': datetime.datetime(2016, 7, 23, 0, 0), 'checkout_date': datetime.datetime(2016, 7, 25, 0, 0), 'is_control': True})]"
        actions = get_booking_dataset(max_actions_per_file=2)[0]
        self.assertEqual(reference_actions, str(actions))


if __name__ == "__main__":
    unittest.main()
        
