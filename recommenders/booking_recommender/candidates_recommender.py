from collections import defaultdict, Counter

from aprec.recommenders.recommender import Recommender

class BookingCandidatesRecommender(Recommender):
    def __init__(self):
        self.user_actions = defaultdict(list)
        self.city_cnt = Counter()
        self.transitions_cnt = defaultdict(Counter)
        self.transitions = defaultdict(list)
        self.booker_trip_country_top = defaultdict(Counter)
        self.booker_country_top = defaultdict(Counter)
        self.trip_country_top = defaultdict(Counter)
        self.booker_country_cnt = Counter()
        self.city_pairs = Counter()
        self.trip_country_cnt = Counter()
        self.booker_trip_country_cnt = Counter()
        self.city_transitions_idx = {}
        self.booker_trip_country_top_idx = {}
        self.trip_country_top_idx = {}
        self.booker_country_top_idx = {}
        self.top_city_idx = {}
        self.city_country_mapping = {}
        self.sample_user_id = None
        self.city_month_cnt = defaultdict(lambda: [0] * 12)
        self.city_year_cnt = defaultdict(lambda: [0]*3)
        self.zero_month_vector = [0]*12
        self.zero_year_vector = [0, 0, 0]

    def add_action(self, action):
        self.user_actions[action.user_id].append(action)
        self.sample_user_id = action.user_id
        self.city_country_mapping[action.item_id] = action.data['hotel_country']

    def rebuild_model(self):
        for user in self.user_actions:
            booker_country = self.user_actions[user][-1].data['booker_country']
            trip_country = self.user_actions[user][-1].data['hotel_country']
            self.booker_country_cnt[booker_country] += 1 
            self.trip_country_cnt[trip_country] += 1
            self.booker_trip_country_cnt[(booker_country, trip_country)] += 1

            for i in range(len(self.user_actions[user]) - 1):
                action = self.user_actions[user][i]
                current_city = self.update_city_stat(action, booker_country, trip_country)

                next_city = self.user_actions[user][i + 1].item_id
                self.transitions_cnt[current_city][next_city] += 1
                for j in range(i + 1, len(self.user_actions[user])):
                    next_city = self.user_actions[user][j].item_id
                    self.city_pairs[(current_city, next_city)] += 1
                    self.city_pairs[(next_city, current_city)] += 1

            action = self.user_actions[user][-1]
            self.update_city_stat(action, booker_country, trip_country)

        for city in self.transitions_cnt:
            self.transitions[city] = self.transitions_cnt[city].most_common()
            for i in range(len(self.transitions[city])):
                next_city, n_transitions = self.transitions[city][i]
                self.city_transitions_idx[(city, next_city)] = (i, n_transitions / self.city_cnt[city])

        for country_trip in self.booker_trip_country_top:
            self.booker_trip_country_top[country_trip] = self.booker_trip_country_top[country_trip].most_common()
            for i in range(len(self.booker_trip_country_top[country_trip])):
                city, cnt = self.booker_trip_country_top[country_trip][i]
                self.booker_trip_country_top_idx[(country_trip, city)] = (i, cnt/ self.booker_trip_country_cnt[country_trip])

        for country in self.trip_country_top:
            self.trip_country_top[country] = self.trip_country_top[country].most_common()
            for i in range(len(self.trip_country_top[country])):
                city, cnt = self.trip_country_top[country][i]
                self.trip_country_top_idx[(country, city)] = (i, cnt/ self.trip_country_cnt[country])

        for country in self.booker_country_top:
            self.booker_country_top[country] = self.booker_country_top[country].most_common()
            for i in range(len(self.booker_country_top[country])):
                city, cnt = self.booker_country_top[country][i]
                self.booker_country_top_idx[(country, city)] = (i, cnt/ self.booker_country_cnt[country])

        for city in self.city_month_cnt:
            city_cnt = self.city_cnt[city]
            for month in range(12):
                self.city_month_cnt[city][month] /= city_cnt

        for city in self.city_year_cnt:
            city_cnt = self.city_cnt[city]
            for year in range(3):
                self.city_year_cnt[city][year] /= city_cnt


        self.top_cities = self.city_cnt.most_common()
        for i in range(len(self.top_cities)):
            city, cnt = self.top_cities[i]
            self.top_city_idx[city] = (i, cnt / len(self.user_actions))

        self.inf = len(self.city_cnt) + 1



        self.booker_country_cnt = dict(self.booker_country_cnt)
        self.booker_country_top = dict(self.booker_country_top)
        self.booker_trip_country_cnt = dict(self.booker_trip_country_cnt)
        self.booker_trip_country_top = dict(self.booker_trip_country_top)
        self.trip_country_cnt = dict(self.trip_country_cnt)
        self.trip_country_top = dict(self.trip_country_top)
        self.city_cnt = dict(self.city_cnt)
        self.transitions = dict(self.transitions)
        self.transitions_cnt = dict(self.transitions_cnt)
        self.city_pairs = dict(self.city_pairs)
        self.user_actions = dict(self.user_actions)
        self.city_month_cnt = dict(self.city_month_cnt)

        trip = self.user_actions[self.sample_user_id]
        features = self.get_candidates_with_features(trip, 1)
        self.n_features = len(features[0][1])

    def update_city_stat(self, action, booker_country, trip_country):
        current_city = action.item_id
        trip_month = action.timestamp.month - 1
        trip_year = action.timestamp.year - 2015
        self.city_month_cnt[current_city][trip_month] += 1
        self.city_year_cnt[current_city][trip_year] += 1
        self.city_cnt[current_city] += 1
        self.booker_trip_country_top[(booker_country, trip_country)][current_city] += 1
        self.booker_country_top[booker_country][current_city] += 1
        self.trip_country_top[trip_country][current_city] += 1
        return current_city

    def recommend_items_by_trip(self, trip, limit):
        trip_cities = [action.item_id  for action in trip]
        trip_candidates = Counter(trip_cities)
        result = list(trip_candidates.most_common())
        already_recommended = set([item for item,score in result])

        if len(trip) > 0:
            last_city = trip_cities[-1]
            booker_country = trip[-1].data['booker_country']
            trip_country = trip[-1].data['hotel_country']

            transitions = self.transitions.get(last_city, ())
            for next_city, cnt in transitions:
                if len(result) >= limit or len(result)>= 150:
                    break
                if next_city not in already_recommended:
                    result.append((next_city, 0))
                    already_recommended.add(next_city)

            for next_city, cnt in self.booker_trip_country_top.get((booker_country, trip_country), ()):
                if len(result) >= limit or len(result) >= 350:
                    break
                if next_city not in already_recommended:
                    result.append((next_city, 0))
                    already_recommended.add(next_city)

            for next_city, cnt in self.trip_country_top.get(trip_country, ()):
                if len(result) >= limit:
                    break
                if next_city not in already_recommended:
                    result.append((next_city, 0))
                    already_recommended.add(next_city)

            for next_city, cnt in self.booker_country_top.get(booker_country, ()):
                if len(result) >= limit:
                    break
                if next_city not in already_recommended:
                    result.append((next_city, 0))
                    already_recommended.add(next_city)

        for next_city, cnt in self.top_cities:
            if len(result) >= limit:
                break
            if next_city not in already_recommended:
                result.append((next_city, 0))
                already_recommended.add(next_city)
        return result

    def get_candidates_with_features(self, trip, limit):
        recs = self.recommend_items_by_trip(trip, limit)
        trip_counter = Counter([action.item_id for action in trip])

        if len(trip) > 0:
            booker_country = trip[-1].data['booker_country']
            trip_country = trip[-1].data['hotel_country']
        else:
            booker_country = ""
            trip_country = ""

        result = []
        for city, _ in recs:
            city_country = self.city_country_mapping.get(city, "")

            candidate_vector = []

            booker_trip_pos, booker_trip_prob = self.booker_trip_country_top_idx.get(((booker_country, trip_country), city), (self.inf, 0.0))
            candidate_vector.append(booker_trip_pos/self.inf)
            candidate_vector.append(booker_trip_prob/self.inf)

            trip_country_pos, trip_country_prob = self.trip_country_top_idx.get((trip_country, city), (self.inf, 0.0))
            candidate_vector.append(trip_country_pos / self.inf)
            candidate_vector.append(trip_country_prob)


            booker_country_pos, booker_country_prob = self.booker_country_top_idx.get((booker_country, city), (self.inf, 0.0))
            candidate_vector.append(booker_country_pos / self.inf)
            candidate_vector.append(booker_country_prob)

            candidate_vector.append(int(city_country == booker_country))
            candidate_vector.append(int(city_country == trip_country))


            #city global popularty
            city_pos, city_pop = self.top_city_idx.get(city, (self.inf, 0.0))
            candidate_vector.append(city_pos / self.inf)

            candidate_vector.append(city_pop)

            #is the city equal to the first city in the trip
            if len(trip) > 0:
                candidate_vector.append(int(city == trip[0].item_id))
            else:
                candidate_vector.append(0)

            #how popular is the city within the trip
            candidate_vector.append(trip_counter[city] / (len(trip) + 0.0001))

            candidate_month_distr = self.city_month_cnt.get(city, self.zero_month_vector)
            candidate_vector += candidate_month_distr

            candidate_year_distr = self.city_year_cnt.get(city, self.zero_year_vector)
            candidate_vector += candidate_year_distr

            city_cnt = self.city_cnt.get(city, 0)

            for i in range(5):
                transition_pos = self.inf
                transition_prob = 0.0
                same_as_trip_city = 0
                same_country = 0
                idx = len(trip) -i - 1
                cos_sim = 0.0
                if idx >= 0:
                    trip_city = trip[idx].item_id
                    transition_pos, transition_prob = self.city_transitions_idx.get((trip_city, city), (self.inf, 0.0))

                    trip_city_cnt = self.city_cnt.get(trip_city, 0)
                    pair_cnt = self.city_pairs.get((trip_city, city), 0)
                    cos_sim = pair_cnt ** 2 / (city_cnt * trip_city_cnt + 0.00001)
                    same_as_trip_city = int(city == trip_city)
                    same_country = int(self.city_country_mapping.get(trip_city, "") == city_country)
                candidate_vector.append(transition_pos / self.inf)
                candidate_vector.append(transition_prob)
                candidate_vector.append(same_as_trip_city)
                candidate_vector.append(same_country)
                candidate_vector.append(cos_sim)
            result.append((city, candidate_vector))
        return result

    def get_next_items(self, user_id, limit, features=None):
        trip = self.user_actions[user_id]
        return self.recommend_items_by_trip(trip, limit)

