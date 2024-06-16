import numpy as np
import pandas as pd  # type: ignore
from dataclasses import dataclass


def read_households_distribution_template(file_name: str,
                                          input_folder: str = "") -> pd.DataFrame:
    """  Метод загружает шаблон с числом людей в домохозяйствах по регионам  """

    # загрузить шаблон, убрать пустые строки и столбцы
    households_distribution_template = pd.read_excel(input_folder + file_name) \
                                         .iloc[4:, 1:8].reset_index(drop=True)
    
    households_distribution_template.columns = pd.Index(["region_type", "1_person", "2_persons",
                                                         "3_persons", "4_persons", "5_persons",
                                                         "6+_persons"])

    # вынести название с областью в отдельный столбец
    households_distribution_template["region"] = \
        np.array(households_distribution_template.region_type)[::3].repeat(3)

    # убрать строки, которые содержат название областей
    households_distribution_template = \
        households_distribution_template[households_distribution_template.region_type.str.contains("пункты")]\
        [["region", "region_type", "1_person", "2_persons", "3_persons", "4_persons", "5_persons", "6+_persons"]]
    
    # убрать лишние переносы строки в названиях областей
    households_distribution_template["region"] = \
        households_distribution_template["region"].apply(lambda x: " ".join(x.split()))
    
    households_distribution_template = households_distribution_template.reset_index(drop=True)
    return households_distribution_template


def read_age_sex_distribution_template(file_name: str,
                                       input_folder: str = "") -> pd.DataFrame:
    """ Метод загружает шаблон со средним по регионам возрастно-половым распределением """
    
    # загрузить шаблон, убрать пустые строки и столбцы
    df_inner = pd.read_excel(input_folder + file_name) \
                 .iloc[5:-1, 1:].reset_index(drop=True)

    df_inner.columns = pd.Index(["age", "men_women_total", "men_total",
                                 "women_total", "men_women_urban", "men_urban",
                                 "women_urban", "men_women_rural", "men_rural",
                                 "women_rural"])

    df_inner["age_group"] = df_inner["age"]

    # перевести значения из процентов в доли
    col = pd.Index(["men_women_total", "men_total",
                    "women_total", "men_women_urban", "men_urban",
                    "women_urban", "men_women_rural", "men_rural",
                    "women_rural"])

    df_inner.loc[:, col] = df_inner.loc[:, col] / 100

    # избавиться от диапазонов вида a-b, продублировав строку b-a+1 раз
    tmp = pd.concat([df_inner.iloc[:-1, :]]*5).reset_index(drop=True)

    tmp["age"] = tmp["age"].apply(lambda x: x.split()[0])
    
    tmp = tmp.sort_values("age") \
             .reset_index(drop=True) \
             .reset_index() \
             .drop(columns=["age"]) \
             .rename(columns={"index": "age"})

    # разделить доли на равные части между каждыми значениями из диапазона
    # и перевести это в доли из процентов
    tmp.loc[:, col] = tmp.loc[:, col] / 5

    # добавить категорию "70 лет и более"
    tmp = pd.concat([tmp, df_inner.iloc[-1:, :]]).reset_index(drop=True)

    # сделать проверку, что контрольная сумма по всем долям равна 1
    if not np.isclose(abs(tmp.loc[:, col].sum(axis=0)).max(), 1):
        raise Exception
    
    return tmp


def read_manufactures_distribution_template(file_name: str,
                                            input_folder: str = "") -> pd.DataFrame:
    """ Функция загружает шаблон распределения предприятий по регионам России """

    # загрузить шаблон, убрать пустые строки и столбцы
    df_inner = pd.read_excel(input_folder + file_name).iloc[5:, 1:]

    df_inner.columns = pd.Index(["Название округа ", "Название области", "Код ОКАТО", "Сельское, лесное хозяйство",
                                 "Добыча полезных ископаемых", "Обрабатывающие производства",
                                 "Обеспечение электрической энергией, газом и паром", "Водоснабжение", "Строительство",
                                 "Торговля оптовая и розничная", "Транспортировка и хранение",
                                 "Деятельность гостиниц и предприятий общественного питания",
                                 "Деятельность в области информации и связи", "Деятельность финансовая и страховая",
                                 "Деятельность по операциям с недвижимым имуществом",
                                 "Деятельность профессиональная, научная и техническая",
                                 "Государственное управление и обеспечение военной безопасности", "Образование",
                                 "Деятельность в области здравоохранения", "Деятельность в области культуры, спорта",
                                 "Предоставление прочих видов услуг",
                                 "Недифференцированная деятельность частных домашних хозяйств",
                                 "Деятельность экстерриториальных организаций и органов"])

    return df_inner


def read_schools_distribution_template(file_name: str,
                                       input_folder: str = "") -> pd.DataFrame:
    """ Функция загружает шаблон распределения школ по регионам России """

    # загрузить шаблон, убрать пустые строки и столбцы
    df_inner = pd.read_excel(input_folder + file_name).iloc[5:, [1, 2, 7, 12, 17]]
    df_inner.columns = pd.Index(["Название округа", "Название области", "Число школ",
                                 "Число обучающихся", "Число обучающихся на одну школу"])
    df_inner["Число обучающихся"] = df_inner["Число обучающихся"] * 1000
    df_inner["Число обучающихся на одну школу"] = df_inner["Число обучающихся на одну школу"] * 1000
    df_inner = df_inner.astype({"Название округа": str,
                                "Название области": str,
                                "Число школ": int,
                                "Число обучающихся": int,
                                "Число обучающихся на одну школу": int})

    return df_inner
























