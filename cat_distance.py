'''
Расчет расстояний для категориальных признаков
Реализованы:
- расчет расстояний методом Говера
- расчет расстояний методом Левенштейна
- расчет расстояний как доля совпашвших слов в тесте

Зависимости:
Levenshtein
numpy
'''

from Levenshtein import distance as lev_dist
import numpy as np

#Расчет расстояний методом Говера
def gover(one_col, weight, **kwargs):
  #Для каждой позичии в векторе
  for ind, value in enumerate(one_col):
    dist_col = [0 if value == row_value else 1 for row_value in one_col]
    dist_col = [v*weight for v in dist_col]
    if ind != 0:
      dist_matrix = np.vstack([dist_matrix, dist_col])
    else:
      dist_matrix = dist_col
  return dist_matrix

#Расчет вектора расстояний ГОВЕРА по одному признаку для одного нового значения
def new_gover(main_col_data, new_col, weight, **kwargs):
  main_col_data = np.append(main_col_data, new_col)
  #Для каждой позичии в векторе
  # #Как связано новое значение в колонке с остальными ранее рассчитанными значениями
  dist_col = [0 if new_col == row_value else 1 for row_value in main_col_data]
  dist_col = [v*weight for v in dist_col]
  return dist_col

#Расчет вектора расстояний (как отношение кол-ва совпавших слов (между новым значением и значением в столбце)
# к колву слов в новом значении)
def text_one_value(value, one_col, weight, **kwargs):
  value_set = set(value.split())
  if len(value_set) == 0:
    dist_col = [0 if len(row_value.split()) == 0 else 1 for row_value in one_col]
  else:
    dist_col = [1 - (len(value_set & set(row_value.split()))/len(value_set)) for row_value in one_col]
  dist_col = [v*weight for v in dist_col]
  return dist_col

#Расчет матрицы расстояний по одному признаку
def dist_matrix_text(one_col, weight, **kwargs):
  #Для каждой позичии в векторе
  for ind, value in enumerate(one_col):
    dist_col = text_one_value(value, one_col, weight, **kwargs)
    if ind != 0:
      dist_matrix = np.vstack([dist_matrix, dist_col])
    else:
      dist_matrix = dist_col
  return dist_matrix

#Дорасчет матрицы расстояний одного признака с учетом добавления нового значения
def new_row_in_text(main_col_data, new_row_data, weight, **kwargs):
  main_col_data = np.append(main_col_data, new_row_data)
  try:
    value_set = set(new_row_data.split())
  except:
    value_set = set(new_row_data)
  if len(value_set) == 0:
    dist_col = [0 if len(row_value.split())== 0 else 1 for row_value in main_col_data]
  else:
    dist_col = [1 - (len(value_set & set(row_value.split()))/len(value_set)) for row_value in main_col_data]
  dist_col = [v*weight for v in dist_col]
  return dist_col

#Расчет матрицы расстояний по одному признаку для одного нового значения
def new_text(main_col_data, new_col, weight, **kwargs):
  main_col_data = np.append(main_col_data, new_col)
  #Для каждой позичии в векторе
  dist_col = new_row_in_text(main_col_data, new_col, weight, **kwargs)
  dist_cols.append(dist_col)
  return dist_cols

#Расстояние Ливенштейна (Levenshtein distance)
#distance between words represents the minimum number of single-character
# edits required to change one word into the other
def levenshtain(one_col, weight, **kwargs):
  n_head = kwargs['obj'].n_head
  #Для каждой позичии в векторе
  for ind, value in enumerate(one_col):
    n_end = len(value) - n_head
    if n_end < 1:
      raise Exception (f'''ValueError. The part of string for Levenstain less then 1. Try to reduce n_head. ''')
    dist_col = []
    #Cравниваем, сходятся ли начальные части с каждым значением в этом же векторе
    dist_matrix = []
    for row_value in one_col:
      if value[:n_head] != row_value[:n_head]:
        dist_col.append(1)
      else:
        #Считаем расстояние для тех позиций, в которых совпали заголовки
        dist_col.append(round(lev_dist(row_value[n_head + 1:], value[n_head + 1:])/n_end, 3))
      # dist_col = [v*weight for v in dist_col]
      dist_matrix.append(dist_col)
  return np.array(dist_matrix)

#Расчет вектора расстояний на базе ЛЕвенштейна по одному признаку для одного нового значения
def new_levenshtain(main_col_data, new_col, weight, **kwargs):
  main_col_data = np.append(main_col_data, new_col)
  n_head = kwargs['obj'].n_head
  n_end = len(new_col) - n_head
  dist_col = []
  #Cравниваем, сходятся ли начальные части с каждым значением в этом же векторе
  for row_value in main_col_data:
    if n_end >=1:
      if new_col[:n_head] != row_value[:n_head]:
        dist_col.append(1)
    #Считаем расстояние для тех позиций, в которых совпали заголовки
    dist_col.append(round(lev_dist(row_value[n_head + 1:], new_col[n_head + 1:])/n_end, 3))
    dist_col = [v*weight for v in dist_col]
  return dist_col