'''
Класс (ChangeText) с методами для предобработки текстовых даннных
Реализованы:
- отчистка от символов и лишних пробелов
- отчистка от повторяющихся символов
- учет предлога нет (если в строке есть частица не ЕЕ можно объединить с соседним словом)
- пересчет строки в строку с нормализованными словами, И без стоп слов
- фильтрация токенов на основании TF-IDF важности

Зависимости:
re
pymorphy3
sklearn.feature_extraction.text.TfidfVectorizer
'''


import re
import pymorphy3
from sklearn.feature_extraction.text import TfidfVectorizer

'''
Класс для Обработки текстовых столбцов
text_for_fill - строка, которой будут заполняться пустые значения
repeat_count - число повторений одного символа, которые будут заменять на единичный символ (aaa -> a) 
connect_ne - надо ли склеивать частицу не с ближайшим правым словом
trash_symbols - строка символов, которые надо удалять из текстовых данных
stop_words - список стоп-слов, которые надо удалять из текстовых данных,
limit_by_tfidf - необходимо ли удалять из строк слова, значимосьб которых по tfidf ниже порога, 
tfidf_trashold  - порог для значимости по tfidf,
save_order - нужно ли сохранять порядок слов после фильтрации по tfidf,
method - метод рассчета расстояний, 
n_head = 0 - число букв, которые сравниваются как корни при расчете рсстояний методом Левенштейна
'''
class ChangeText():
  def __init__(self, text_for_fill = 'empty', repeat_count = 2, connect_ne = False,
               trash_symbols = '!@#$%^&*()-', stop_words = [],
               limit_by_tfidf = False, tfidf_trashold = 0.3, save_order = False,
               method = 'gover', n_head = 0):
    self.repeat_count = repeat_count
    self.trash_symbols = trash_symbols
    self.stop_words = stop_words
    self.limit_by_tfidf = limit_by_tfidf
    self.tfidf_trashold = tfidf_trashold
    self.save_order = save_order
    self.text_for_fill = text_for_fill
    self.method = method
    self.n_head = n_head
#!!! сеттер для n_head

  #Сеттер и геттер для repeat_count(число повторений букв, которые надо переделать)
  @property
  def repeat_count(self):
    return self._repeat_count

  @repeat_count.setter
  def repeat_count(self, repeat_count):
    #Проверка, что передано целое число
    if type(repeat_count) != int:
      raise Exception (f'''TypeError. Wrong type for repeat_count. Expected int. Got: {type(repeat_count)}.''')
    elif repeat_count < 2:
      raise Exception (f'''ValueError.  Repeat_count must be greate then 1.  Got: {repeat_count}.''')
    else:
      self._repeat_count = repeat_count

  #Сеттер и геттер для repeat_count(число повторений букв, которые надо переделать)
  @property
  def trash_symbols(self):
    return self._trash_symbols

  @trash_symbols.setter
  def trash_symbols(self, trash_symbols):
    #Проверка, что передано целое число
    if type(trash_symbols) != str:
      raise Exception (f'''TypeError. Wrong type for trash_symbols. Expected str. Got: {type(trash_symbols)}.''')
    else:
      self._trash_symbols =  trash_symbols

  #Сеттер и геттер для limit_by_tfidf(число повторений букв, которые надо переделать)
  @property
  def limit_by_tfidf(self):
    return self._limit_by_tfidf

  @limit_by_tfidf.setter
  def limit_by_tfidf(self, limit_by_tfidf):
    #Проверка, что передано целое число
    if limit_by_tfidf not in (False, True):
      raise Exception (f'''ValueError. Wrang value for limit_by_tfidf. Expected False or True. Got: {limit_by_tfidf}.''')
    else:
      self._limit_by_tfidf =  limit_by_tfidf

  #Сеттер и геттер для save_order
  @property
  def save_order(self):
    return self._save_order

  @save_order.setter
  def save_order(self, save_order):
    #Проверка, что передано целое число
    if save_order not in (False, True):
      raise Exception (f'''ValueError. Wrang value for save_order. Expected False or True. Got: {save_order}.''')
    else:
      self._save_order =  save_order

  #Сеттер и геттер для tfidf_trashold(число повторений букв, которые надо переделать)
  @property
  def tfidf_trashold(self):
    return self._tfidf_trashold

  @tfidf_trashold.setter
  def tfidf_trashold(self, tfidf_trashold):
    #Проверка, что передано число
    if type(tfidf_trashold) not in (float, int):
      raise Exception (f'''TypeError. Wrong type for tfidf_trashold. Expected int, float. Got: {type(tfidf_trashold)}.''')
    elif tfidf_trashold >=1:
      raise Exception (f'''ValueError. tfidf_trashold must be lower then 1.  Got: {tfidf_trashold}.''')
    elif tfidf_trashold <0:
      raise Exception (f'''ValueError. tfidf_trashold must be higher or equal 0.  Got: {tfidf_trashold}.''')
    else:
      self._tfidf_trashold =  tfidf_trashold

#Сеттер и геттер для заполнителя пустых значений
  @property
  def text_for_fill(self):
    return self._text_for_fill

  @text_for_fill.setter
  def text_for_fill(self, text_for_fill):
    #Проверка, что передан корректный аргумент
    if type(text_for_fill) != str:
      raise Exception (f'''ValueError.  Value "text_for_fill" must be string.  Got: {type(text_for_fill)}.''')
    self._text_for_fill = text_for_fill

  #Отчистка от символов и лишних пробелов
  def clean_text(self, text_list):
    #Шаблон регулярного выражения для удаления заданных символов
    pattern = f"[{self.trash_symbols}]+"
    text_list = [re.sub(pattern, ' ', text) for text in text_list]
    text_list = [' '.join(text.split()) for text in text_list]
    return text_list

  #Отчистка от повторяющихся символов
  def delete_repeat_letters(self, text_list):
    print(text_list)
    #Ищем символы, которые повторяются repeat_count и более раз подряд
    pattern = re.compile(r'(\w)(\1{'+f'{self.repeat_count - 1}'+r',})')
    #заменяем на один раз
    repl = r'\1'
    #Заменяем повторы
    text_list = [re.sub(pattern, repl, str(text)) for text in text_list]
    text_list = [' '.join(text.split()) for text in text_list]
    return text_list

  #Учет предлога нет
  #Если в строке есть частица не объединяем ее с соседним словом
  def connect_ne(self, text_list):
    pattern = re.compile(r'([не])(\s)(\S+)')
    repl = r'\1\3'
    text_list = [re.sub(pattern, repl, text) for text in text_list]
    text_list = [' '.join(text.split()) for text in text_list]
    return text_list

  #Пересчитывает строки в строку с нормализованными словами, И без стоп слов
  def norm_text(self, text_list):
    #Переводим в нижний регистр
    text_list = [text.lower() for text in text_list]
    #Создаем объект языковой модели
    morph = pymorphy3.MorphAnalyzer()
    #Список нормализованных слов для каждой строки
    new_text_list = []
    for text in text_list:
      word_list = [morph.normal_forms(word)[0] for word in text.split() if word not in self.stop_words]
      #Нормализованная строка
      text = " ".join(word_list)
      new_text_list.append(text)
    return new_text_list

  #Расчет матрицы TF-IDF (отражает важность слова внутри строки с учетом
  #его частоты появления внутри всего анализируемого текста)
  def calc_tfidf(self, text_list):
    # Создание объекта TfidfVectorizer
    try:
      tfidf_vectorizer = self.tfidf_obj
    except:
      tfidf_vectorizer = TfidfVectorizer()
      tfidf_vectorizer = tfidf_vectorizer.fit(text_list)

    #Если надо сохранить порядок слов
    if self.save_order:
      #Фиксируем порядок, который изначально был в строке
      order_text_list = [[word for word in text.split()] for text in text_list]

    # Применение TF-IDF к текстовым данным
    tfidf_matrix = tfidf_vectorizer.transform(text_list)

    #Фиксируем tfidf объект для последующих расчетов добавляемых строк
    self.tfidf_obj = tfidf_vectorizer

    #Отбираем только тех, у которых значимость больше trashhold
    #Список для сбора финальных строк
    finall_text_list = []
    #Все слова, для которых посчитана матрица
    all_words = tfidf_vectorizer.get_feature_names_out()
    #Итерируем по матрице (каждый шаг - список значимостей для строки из начального списка строк)
    for row_ind, row_tfidf in enumerate(tfidf_matrix):
      #Список значимостей для каждого слова из all_words в применении к строке
      tfidf_scors = row_tfidf.toarray()[0]
      #Ищем слова, которые значимы в рассматриваемой строке
      need_words = [all_words[word_ind] for word_ind, score in enumerate(tfidf_scors) if score >= self.tfidf_trashold]
      #Если без сохранения порядка следования
      if not self.save_order:
        finall_words = ' '.join(need_words)
      else:
        #Расставляем слова в нужном порядке
        finall_words = ' '.join([word for word in order_text_list[row_ind] if word in need_words])
      finall_text_list.append(finall_words)
    return finall_text_list

  #Предобработки текстовых столбцов
  def prepare_text(self, text_list):
    text_list = self.delete_repeat_letters(text_list)
    text_list = self.clean_text(text_list)
    if self.connect_ne:
      text_list = self.connect_ne(text_list)
    text_list = self.norm_text(text_list)
    if self.limit_by_tfidf:
      text_list = self.calc_tfidf(text_list)
    return text_list
