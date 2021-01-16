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



def get_booking_dataset_one_file(filename, is_testset=False, max_actions=None):
    actions = []
    submit_actions = []
    with open(filename) as input_file:
        input_file.readline()
        next_line = input_file.readline()
        while(len(next_line) != 0):
            parts = split_csv_string(next_line)
            try:
                if not is_testset:
                    _, user_id, checkin_str, checkout_str, city_id,\
                            device_class, affiliate_id,\
                            booker_country, hotel_country, utrip_id = parts
                else:
                    user_id, checkin_str, checkout_str, device_class, affiliate_id,booker_country,utrip_id,row_num,\
                                total_rows,city_id,hotel_country = parts

            except Exception as ex:
                raise Exception(f"incorrect line: {next_line}")
            date_format = "%Y-%m-%d"
            checkin = datetime.strptime(checkin_str, date_format)
            checkout = datetime.strptime(checkout_str, date_format)
            action = Action(user_id = utrip_id, item_id = city_id, timestamp=checkin, data = {'user_id': user_id,
                                                                          'device_class': device_class,
                                                                          'affiliate_id': affiliate_id,
                                                                          'hotel_country': hotel_country,
                                                                          'booker_country': booker_country,
                                                                          'checkin_date': checkin,
                                                                          'checkout_date': checkout,
                                                                          'is_control': is_testset
                                                                         })
            if action.item_id == "0":
                submit_actions.append(action)
            else:
                actions.append(action)
            if max_actions is not None and len(actions) >= max_actions:
                break
            next_line = input_file.readline()
    return actions, submit_actions

def get_booking_dataset(train_filename, test_filename, max_actions_per_file=None):
    train_actions, _ = get_booking_dataset_one_file(train_filename, is_testset=False,
                                                 max_actions=max_actions_per_file)
    test_actions, submit_actions = get_booking_dataset_one_file(test_filename, is_testset=True,
                                                 max_actions=max_actions_per_file)
    return train_actions + test_actions, submit_actions


