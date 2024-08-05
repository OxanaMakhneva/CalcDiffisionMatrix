import pandas as pd
from pathlib import Path
from corr_model import *
import numpy as np

# params_cat_1 = {'repeat_count': 2, 'connect_ne': False, 'method': 'levenshtain',
#                'trash_symbols': '!@#$%^&*()-', 'limit_by_tfidf': False, 'save_order': False}
#
# params_cat_2 = {'repeat_count': 2, 'connect_ne': False, 'method': 'text',
#                'trash_symbols': '!@#$%^&*()-', 'limit_by_tfidf': False, 'save_order': False, 'n_head': 3}
#
# params_text = {'repeat_count': 2, 'connect_ne': False, 'method': 'text',
#                'trash_symbols': '!@#$%^&*()-', 'tfidf_trashold': 0.3, 'limit_by_tfidf': True, 'save_order': False}
#
# params_num = {'method': 'absolute', 'num_for_fill': 'mean'}

# data = pd.DataFrame({'text_text': ['повар не варит компот', 'надо повар не варить компот из яблок', 'миша яблоко пошел спасть с котом'],
#                      'gender': ['M', 'M', 'W'],
#                      'age': [20,20,30],
#                      'iq': [1,2,3]})
# q = MyCorr(data, cat_cols = {'gender': params_cat_1, 'text_text': params_cat_2},
#                  num_cols = {'age': params_num, 'iq': params_num})

# data = pd.DataFrame({'cats_1': ['стол','стул','столб'],
#                      'cats_2': ['боб','бос','cтолб']})
# print(data)
# q = MyCorr(data, cat_cols = {'cats_1': {'repeat_count': 2, 'connect_ne': False, 'method': 'gover','trash_symbols': '!@#$%^&*()-', 'limit_by_tfidf': False, 'save_order': False},
#                              'cats_2': {'repeat_count': 2, 'connect_ne': False, 'method': 'levenshtain','trash_symbols': '!@#$%^&*()-', 'limit_by_tfidf': False, 'save_order': False}
#                              })


# q.dx = q.prepare_cols(q.dx)
# dist_1 = q.calc_dist_matrix()
# print(dist_1)
#
# new_row = pd.DataFrame({'cats_1': ['a'], 'cats_2': ['бос']})
# new_row = q.prepare_cols(new_row)
# print(new_row)
# dist_2 = q.calc_dist_matrix(new_row)
# print(dist_2)


params_cat_1 = {'repeat_count': 2, 'connect_ne': False, 'method': 'levenshtain',
               'trash_symbols': '!@#$%^&*()-', 'limit_by_tfidf': False, 'save_order': False}

params_cat_2 = {'repeat_count': 2, 'connect_ne': False, 'method': 'gover',
               'trash_symbols': '!@#$%^&*()-', 'limit_by_tfidf': False, 'save_order': False, 'n_head': 3}

params_text = {'repeat_count': 2, 'connect_ne': False, 'method': 'text',
               'trash_symbols': '!@#$%^&*()-', 'tfidf_trashold': 0.3, 'limit_by_tfidf': True, 'save_order': False}

params_num = {'method': 'absolute', 'num_for_fill': 'mean'}

data = pd.read_csv(Path('C:/Users/OkCu/Desktop/rrr.csv'), sep = ';', encoding = 'cp1251')

data.columns = data.columns.str.lower()
data = data[['title', 'original title', 'actors', 'type', 'year', 'genres', 'countries', 'runtime', 'age limit',
             'directors', 'audience']]

def calc_year(x):
    if re.search('-', str(x)):
        years_list = [int(y) for y in str(x).split('-')]
        return round(np.mean(years_list))
    else:
        return int(x)

data.year = data.year.agg(calc_year)


# print(data.sample(2))
# print(data.columns)
# print(data.info())

q = MyCorr(data, cat_cols = {'type': params_cat_2,
                             'original title': params_text,
                             'actors': params_text,
                             'type': params_text,
                             'genres': params_text,
                             'countries': params_text,
                               'directors': params_text},
                  num_cols = {'audience': params_num,
                              'year': params_num,
                              'runtime': params_num,
                              'age limit': params_num})
q.dx = q.prepare_cols(q.dx)
dist_1 = q.calc_dist_matrix()
print(dist_1)



# new_row = pd.DataFrame({'text_text': ['повар не варит компот суп салат'],
#                         'gender': ['M'],'age': [20],'iq': [1]})
# new_row = q.prepare_cols(new_row)
# dist_2 = q.calc_dist_matrix(new_row)
# print(dist_2)