'''
Класс(MinMaxSc) для масштабирования методом минимакс
Класс (ChangeNum) с методами для предобработки количественнных данных
Реализованы:
- заполнение пустых значений (заглущка, медиана, среднее значение)
- масштабирование методом минимакс

Зависимости:
numpy
'''

import numpy as np

class MinMaxSc():
  def __init__(self):
    self.is_fit = False

  def fit(self, matrix):
    self.min = matrix.min()
    self.max = matrix.max()
    return self

  def transform(self, matrix):
    if self.min != self.max:
      return (matrix - self.min) / (self.max - self.min)
    else:
      return (matrix) / matrix[0]

'''
Класс для Обработки численных столбцов
- num_for_fill - тип заполнения пустых значений
- method - метод расчета расстояний для численнных признаков
- patch - число, которое надо использовать как заплатку если выбран num_for_fill = 'patch'
'''
class ChangeNum():
  def __init__(self, num_for_fill = 'median', method = 'abs', patch = -1):
    self.num_for_fill = num_for_fill
    self.method = method
    self.patch = patch

  @property
  def num_for_fill(self):
    return self._num_for_fill

  @num_for_fill.setter
  def num_for_fill(self, num_for_fill):
    #Проверка, что передан корректный аргумент
    if num_for_fill not in ('median', 'mean', 'patch'):
      raise Exception (f'''ValueError.  Value "num_for_fill" must be one of the next ('median', 'mean', 'patch').  Got: {num_for_fill}.''')
    self._num_for_fill = num_for_fill

  @property
  def patch(self):
    return self._patch

  @patch.setter
  def patch(self, patch):
    #Проверка, что передано целое число
    if type(patch) not in (int, float) :
      raise Exception (f'''TypeError. Wrong type for patch. Expected int or float. Got: {type(patch)}.''')
    self._patch = patch

  @property
  def method(self):
    return self._method

  @method.setter
  def method(self, method):
    #Проверка, что передано целое число
    if method not in ('absolute', 'evclid'):
      raise Exception (f'''ValueError.  Value "method" must be one of the next ('absolute', 'evclid').  Got: {method}.''')
    self._method = method

  #Предобработка численных столбцов
  def prepare_num(self, col_data):
    col_data = self.fill_patch(col_data)
    col_data = self.norm_num(col_data)
    return col_data

#Заполнение пустых значений
  def fill_patch(self, col_data):
    if self.num_for_fill == 'mean':
      patch = np.mean(col_data)
    elif self.num_for_fill == 'median':
      patch = np.median(col_data)
    else:
      patch = self.patch
    col_data = np.nan_to_num(col_data, nan=patch)
    return col_data

#Масштабирование методом минимакс
  def norm_num(self, col_data):
    #Создание объекта MinMaxSc
    try:
      mmsc = self.mmsc_obj
    except:
      mmsc = MinMaxSc()
      mmsc = mmsc.fit(col_data)
    # Применение MinMaxSc к текстовым данным
    norm_data = mmsc.transform(col_data)
    #Фиксируем MinMaxSc объект для последующих расчетов добавляемых строк
    self.mmsc_obj = mmsc
    return norm_data
