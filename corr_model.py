import pandas as pd
from cat_model import *
from num_model import *
from text_model import *


class MyCorr():
  def __init__(self, dx, num_cols = {}, cat_cols = {}, weights = {}):

    #Объявляем список функций, которые могут быть выбраны для расчета расстояний
    self.dist_funcs = {('main', 'gover'): gover, ('main', 'text'): dist_matrix_text,
                  ('main', 'absolute'): absolute, ('main', 'levenshtain'): levenshtain,
                  ('new', 'levenshtain'): new_levenshtain,
                   ('new', 'gover'): new_gover, ('new', 'text'): new_text,
                  ('new', 'absolute'): new_absolute}

    #Проверяем что список с количественными столбцами не совпадают с категориальными если они уже заданы
    commons = set(cat_cols.keys())&set(num_cols.keys())
    if len(commons) != 0:
      raise Exception (f'''ValueError. There are duplicated values in num and cat columns  {commons}''')

    self.dist_matrixes = {}
    self.mmsc_dict = {}

    #Дф через сеттер
    self.dx = dx

    #Список объектов обработки
    self._change_objs = {'num': {col: ChangeNum(**params) for col, params in num_cols.items()},
                         'cat': {col: ChangeText(**params) for col, params in cat_cols.items()}}

    #Расчет количественных столбцов
    self.num_cols = num_cols
    #Расчет категориальных столбцов
    self.cat_cols = cat_cols
    # #Назначение весов через сеттер
    self.weights = weights

  #Сеттер и геттер для ДФ
  @property
  def cat_objs(self, col):
    return self._cat_objs

  #Сеттер и геттер для ДФ
  @property
  def dx(self):
    return self._dx

  @dx.setter
  def dx(self, dx):
    #Проверка, что передан DataFrame
    if type(dx) != pd.core.frame.DataFrame:
      raise Exception (f'''TypeError. Wrong type for DF. Expected pd.core.frame.DataFrame. Got: {type(dx)}.''')
    #Проверка, что ДФ не пустой
    if dx.shape[1] == 0:
      raise Exception (f'''ValueError. The shape of DF must be greate 0''')
    self._dx = dx

  #Сеттер и геттер для списка числовых признаков
  @property
  def num_cols(self):
    return self._num_cols

  @num_cols.setter
  def num_cols(self, num_cols):
    #Проверяем, что все заданные колонки есть в ДФ
    df_cols = self.dx.columns
    for col in num_cols.keys():
      if col not in df_cols:
        raise Exception (f'''ValueError. There is not column {col} in DF''')
    #Отбираем столбцы, у которых тип не относится к количественным
    cat_cols = list(self.dx.select_dtypes(include=['object']).columns)
    cat_to_num = [col for col in num_cols if col in cat_cols]
    #Пробуем преобразовать данные к численному формату
    for col in cat_to_num:
      try:
        self.dx[cat_to_num] = self.dx[cat_to_num].astype('float')
      except:
        raise Exception (f'''ValueError. Can't change type to numeric for column {col}''')
    self._num_cols = list(num_cols.keys())

  #Геттер для списка категориальных признаков
  @property
  def cat_cols(self):
    return self._cat_cols

  @cat_cols.setter
  def cat_cols(self, cat_cols):
    #Проверяем, что все заданные колонки есть в ДФ
    df_cols = list(self.dx.columns)
    for col in cat_cols.keys():
      if col not in df_cols:
        raise Exception (f'''ValueError. There is not column {col} in DF''')
    #Отбираем столбцы, у которых тип не относится к object
    num_cols = list(self.dx.select_dtypes(include=['int', 'float']).columns)
    num_to_cat = [col for col in cat_cols.keys() if col in num_cols]
    #Пробуем преобразовать данные к строковому формату
    for col in num_to_cat:
      try:
        self.dx[num_to_cat] = self.dx[num_to_cat].astype('str')
      except:
        raise Exception (f'''ValueError. Can't change type to str for column {col}''')
    self._cat_cols = list(cat_cols.keys())

  #Сеттер и геттер для весов
  @property
  def weights(self):
    return self._weights

  @weights.setter
  def weights(self, weights):
    #Проверка, проверка, что весовые коэффициенты заданы корректно
    if not weights:
      weights = {key: 1/(len(self.cat_cols)+len(self.num_cols)) for key in self.cat_cols+self.num_cols}
      self.is_null_weights = True
    else:
      if self.is_null_weights == True:
        weights = {key: 1/(len(self.cat_cols)+len(self.num_cols)) for key in self.cat_cols+self.num_cols}
      #Проверка на типы
      try:
        weights = {key: float(value) for key, value in weights.items()}
      except:
        raise Exception (f'''TypeError. Wrong type for values in waights-dict. Expected float. Got: {set([type(w) for w in weights.values()])}.''')
      #Проверка на длину
      if len(weights) != (len(self.cat_cols)+len(self.num_cols)):
        raise Exception (f'''ValueError. Len of waights-dict ({len(weights)}) does't equal the amount of features ({self.dx.shape[1]}).''')
      #Проверка что названия признаков совпадаб для весов и для списков
      if set(weights.keys()) != set(self.cat_cols + self.num_cols):
        raise Exception (f'''ValueError. Names of columns in waights-dict does't equal to the names in cat and num cols.''')
      if sum(weights.values()) != 1:
        raise Exception (f'''ValueError. Sum of waights ({sum(weights.values())}) does't equal 1.''')
    #Записываем получившиеся веса в атрибут класса
    self._weights = weights

  def set_cat_obj(self, col, cat_obj):
    self._change_objs['cat'] = {**self._change_objs['cat'], **{col: cat_obj}}

  def get_cat_obj(self, col):
    return self._change_objs['cat'][col]

  @property
  def cat_objs(self):
    return self._change_objs['cat']

  def set_num_obj(self, col, num_obj):
    self._num_objs = {**self._change_objs['num'], **{col: num_obj}}

  def get_num_obj(self, col):
    return self._change_objs['num'][col]

  @property
  def num_objs(self):
    return self._change_objs['num']

  #Порядок обработки текстовых столбцов
  def prepare_cols(self, df = pd.DataFrame()):
    #Если это первичное преобразование
    if df.shape[0] == 0:
      df = self.dx
    #Обработка категориальных данных
    for col in self.cat_cols:
      #Извлекаем объект - преобразователь текстовых данных
      cat_obj = self.get_cat_obj(col)
      df[col] = cat_obj.prepare_text(df[col].values)
      #Обновляем данные по объекту ChangeText в объекте MyCorr
      self.set_cat_obj(col, cat_obj)
    #Обработка количественых данных
    for col in self.num_cols:
      #Извлекаем объект - преобразователь текстовых данных
      num_obj = self.get_num_obj(col)
      df[col] = num_obj.prepare_num(df[col].values)
      #Обновляем данные по объекту ChangeText в объекте MyCorr
      self.set_num_obj(col, num_obj)
    return df

#qwe

  def calc_new_dist_matrix(self, col_name, dist_col):
    dist_matrix = self.dist_matrixes[col_name]
    #Присоединяем строку снизу
    dist_matrix = np.vstack([dist_matrix, dist_col[:-1]])
    #Присоединяем 0 в конец строки
    # dist_col = dist_col
    #Присоединяем столбец справа
    dist_matrix = np.vstack([dist_matrix.T, dist_col]).T
    return dist_matrix

  # def add_dist_col_in_matrix(self, col_name, dist_cols):
  #   dist_matrix = self.dist_matrixes[col_name]
  #   #Присоединяем строку снизу
  #   dist_matrix = np.vstack([dist_matrix, dist_col])
  #   #Присоединяем 0 в конец строки
  #   dist_col = dist_col + [0]
  #   #Присоединяем столбец справа
  #   dist_matrix = np.vstack([dist_matrix.T, dist_col]).T
  #   return dist_matrix

  def calc_dist_matrix(self, df_new = pd.DataFrame()):
    finall_result_dist_cols = {}
    if df_new.shape[0] != 0:
      df_type = 'new'
      for q, new_row in df_new.iterrows():
        result_dist_cols = {}
        for col in self.cat_cols:
          method = self.get_cat_obj(col).method
          dist_func = self.dist_funcs[(df_type, method)]
          dist_cols = dist_func(self.dx[col].values, new_row.loc[col], self.weights[col], **{'obj': self.cat_objs[col]})
          result_dist_cols[col] = dist_cols
          dist_matrixes = self.calc_new_dist_matrix(col, dist_cols)
          self.dist_matrixes[col] = dist_matrixes
        for col in self.num_cols:
          method = self.get_num_obj(col).method
          dist_func = self.dist_funcs[(df_type, method)]
          dist_cols = dist_func(self.dx[col].values, new_row.loc[col], self.weights[col], **{'obj': self.num_objs[col]})
          result_dist_cols[col] = dist_cols
          dist_matrixes = self.calc_new_dist_matrix(col, dist_cols)
          self.dist_matrixes[col] = dist_matrixes
        #Добавляем новую строку к данным в основном ДФ
        finall_result_dist_cols[q] = result_dist_cols
        self.dx = pd.concat([self.dx, new_row.to_frame().T])
    else:
      df_type = 'main'

      for col in self.cat_cols:
        method = self.get_cat_obj(col).method
        dist_func = self.dist_funcs[(df_type, method)]
        self.dist_matrixes[col] = dist_func(self.dx[col].values, self.weights[col], **{'obj': self.cat_objs[col]})
      for col in self.num_cols:
        method = self.get_num_obj(col).method
        dist_func = self.dist_funcs[(df_type, method)]
        self.dist_matrixes[col] = dist_func(self.dx[col].values, self.weights[col],  **{'obj': self.num_objs[col]})
    finall_matrix = self.calc_finall_matrix()
    return {'finall_matrix':finall_matrix, 'dist_cols': finall_result_dist_cols}

  def calc_finall_matrix(self):
    for ind, col in enumerate(self.num_cols + self.cat_cols):
      if ind == 0:
        finall_matrix = self.dist_matrixes[col]
      else:
        finall_matrix = finall_matrix + self.dist_matrixes[col]
    return finall_matrix


