params_cat_1 = {'repeat_count': 2, 'connect_ne': False, 'method': 'levenshtain',
               'trash_symbols': '!@#$%^&*()-', 'limit_by_tfidf': False, 'save_order': False}

params_cat_2 = {'repeat_count': 2, 'connect_ne': False, 'method': 'text',
               'trash_symbols': '!@#$%^&*()-', 'limit_by_tfidf': False, 'save_order': False, 'n_head': 3}

params_text = {'repeat_count': 2, 'connect_ne': False, 'method': 'text',
               'trash_symbols': '!@#$%^&*()-', 'tfidf_trashold': 0.3, 'limit_by_tfidf': True, 'save_order': False}

params_num = {'method': 'absolute', 'num_for_fill': 'mean'}

data = pd.DataFrame({'text_text': ['повар не варит компот', 'надо повар не варить компот из яблок', 'миша яблоко пошел спасть с котом'],
                     'gender': ['M', 'M', 'W'],
                     'age': [20,20,30],
                     'iq': [1,2,3]})

q = MyCorr(data, cat_cols = {'gender': params_cat_1, 'text_text': params_cat_2},
                 num_cols = {'age': params_num, 'iq': params_num})

q.dx = q.prepare_cols(q.dx)
dist_1 = q.calc_dist_matrix()

new_row = pd.DataFrame({'text_text': ['повар не варит компот суп салат'],
                        'gender': ['M'],'age': [20],'iq': [1]})
new_row = q.prepare_cols(new_row)
dist_2 = q.calc_dist_matrix(new_row)