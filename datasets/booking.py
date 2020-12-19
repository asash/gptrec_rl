from aprec.api.action import Action
from datetime import datetime

def split_csv_string(string):
    parts = []
    current_pos = 0
    state = 'unquoted'
    current_str = ""
    for i in range(len(string.strip())):
        if state == 'unquoted':
            if string[i] == '"':
                state = 'quoted'
            elif string[i] == ',':
                parts.append(current_str)
                current_str = ""
            else:
                current_str += string[i]
        else:
            if string[i] == '"':
                state = "unquoted"
            else:
                current_str += string[i]
    parts.append(current_str)
    return parts



def get_booking_dataset(filename):
    actions = []
    with open(filename) as input_file:
        first_line = input_file.readline()
        next_line = input_file.readline()
        while(len(next_line) != 0):
            parts = split_csv_string(next_line)
            try:
                _, user_id, checkin_str, checkout_str, city_id,\
                        device_class, affiliate_id,\
                        booker_country, hotel_country, utrip_id = parts
            except Exception as ex:
                raise Exception(f"incorrect line: {next_line}")
            date_format = "%Y-%m-%d"
            checkin = datetime.strptime(checkin_str, date_format)
            checkout = datetime.strptime(checkout_str, date_format)
            action = Action(user_id = user_id, item_id = city_id, timestamp=checkin, data = {'trip_id': utrip_id,
                                                                          'device_class': device_class,
                                                                          'affiliate_id': affiliate_id,
                                                                          'hotel_country': hotel_country,
                                                                          'booker_country': hotel_country,
                                                                          'checkin_date': checkin,
                                                                          'checkout_date': checkout
                                                                         })
            actions.append(action)
            next_line = input_file.readline()
    return actions
