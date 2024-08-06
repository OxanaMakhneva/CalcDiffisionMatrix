'''
Расчет расстояний для числовых признаков
Реализованы:
- расчет расстояний методом абсолютных значений

Зависимости:
numpy
'''

import numpy as np

#Метод для расчета расстояния в количественно признаке
def absolute(one_col, weight, **kwargs):
  #Для каждой позичии в векторе
  for ind, value in enumerate(one_col):
    dist_col = (np.absolute(one_col - value).round(3))
    dist_col = [v*weight for v in dist_col]
    if ind != 0:
      dist_matrix = np.vstack([dist_matrix, dist_col])
    else:
      dist_matrix = dist_col
  return dist_matrix

#Расчет матрицы расстояний по одному признаку для одного нового значения
def new_absolute(main_col_data, new_col, weight, **kwargs):
  main_col_data = np.append(main_col_data, new_col)
  #Как связано новое значение в колонке с остальными ранее рассчитанными значениями
  dist_col = np.absolute(new_col - main_col_data).round(3)
  dist_col = [v*weight for v in dist_col]
  return dist_col