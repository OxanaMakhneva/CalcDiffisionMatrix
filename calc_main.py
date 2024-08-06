'''
Пример расчета матрицы расстояний на основании файла с данными о фильмах
'''

import pandas as pd
from pathlib import Path
from diff_model import *
import numpy as np


params_cat_1 = {'repeat_count': 2, 'connect_ne': False, 'method': 'levenshtain',
               'trash_symbols': '!@#$%^&*()-', 'limit_by_tfidf': False, 'save_order': False}
params_cat_2 = {'repeat_count': 2, 'connect_ne': False, 'method': 'gover',
               'trash_symbols': '!@#$%^&*()-', 'limit_by_tfidf': False, 'save_order': False, 'n_head': 3}
params_text = {'text_for_fill': 'empty', 'repeat_count': 2, 'connect_ne': False, 'method': 'text',
               'trash_symbols': '!@#$%^&*()-', 'tfidf_trashold': 0.1, 'limit_by_tfidf': False, 'save_order': False}
params_num = {'method': 'absolute', 'num_for_fill': 'patch', 'patch': -1}

#Считывание и отбор данных, на основании которых будет считаться матрица
data = pd.read_excel(Path('C:/Users/okcu/Desktop/DF/report.xlsx'))
data.columns = data.columns.str.lower()
data = data[['title', 'original title', 'actors', 'type', 'year', 'genres', 'countries', 'runtime', 'age limit',
             'directors', 'audience']]
def calc_year(x):
    if re.search('-', str(x)):
        years_list = [int(y) for y in str(x).split('-')]
        return round(np.mean(years_list))
    else:
        return int(x)

data.year = data.year.transform(calc_year)


#Определение класса для расчета расстояний
q = MyDiff(data, cat_cols = {'type': params_cat_2,
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
#Предобработка данных
q.dx = q.prepare_cols(q.dx)
#Расчет матрицы расстояний
dist_1 = q.calc_dist_matrix()
print(dist_1)
print(q.dx)


# new_row = pd.DataFrame({'text_text': ['повар не варит компот суп салат'],
#                         'gender': ['M'],'age': [20],'iq': [1]})
# new_row = q.prepare_cols(new_row)
# dist_2 = q.calc_dist_matrix(new_row)
# print(dist_2)