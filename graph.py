import numpy as np
import pandas as pd  # type: ignore
from typing import Tuple, Literal, Union
from scipy.sparse import lil_matrix
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import networkx as nx
import datetime
from scipy.interpolate import interp1d
from tqdm.notebook import tqdm
from dataclasses import dataclass
from scipy.integrate import odeint

from scipy import stats




@dataclass
class MyPopulation:

    def __init__(self, random_seed = 42):
        self.households_distribution_template = None
        self.age_sex_distribution_template = None
        self.manufactures_distribution_template = None
        self.schools_distribution_template = None
        self.population = None                  # в этот атрибут записывается результат метода create_population
        self.population_total = None            # в этот атрибут записывается результат метода generate_total_population
        self.population_total_grouped = None    # это сгруппированная популяция population_total для графиков
        self.display_status = True
        self.random_seed = random_seed
        self.population_nodes_degrees = None
        np.random.seed(self.random_seed)

    def read_households_distribution_template(self, file_name: str,
                                              input_folder: str = "") -> int:
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
            households_distribution_template[households_distribution_template.region_type.str.contains("пункты")] \
                [
                ["region", "region_type", "1_person", "2_persons", "3_persons", "4_persons", "5_persons", "6+_persons"]]

        # убрать лишние переносы строки в названиях областей
        households_distribution_template["region"] = \
            households_distribution_template["region"].apply(lambda x: " ".join(x.split()))

        households_distribution_template = households_distribution_template.reset_index(drop=True)
        self.households_distribution_template = households_distribution_template

        return 0

    def plot_sir_model_curve(self,
                             population_size: int = 1000,
                             beta: np.ndarray = np.array([0.2]),  # коэффициент контактов [0, 1]
                             gamma: float = 1. / 10,  # коэффициент выздоровления в 1/ (число дней)
                             plot_s: bool = True,  # рисовать ли кривую для восприимчивых
                             plot_i: bool = True,  # рисовать ли кривую для инфицированных
                             plot_r: bool = True,  # рисовать ли кривую для выздоровевших
                             n_days: int = 160,  # промежуток дней, за которых рисовать график
                             i0: int = 1,  # начальные значения для зараженных и выздоровевших
                             r0: int = 0,
                             use_generated_population_nodes_degrees = False) -> int:

        # чисор восприимчивых людей
        s0 = population_size - i0 - r0
        t = np.linspace(0, n_days, n_days)

        # Система дифференциальных уравнения SIR
        def deriv(y, t, N, beta, gamma):
            S, I, R = y
            dSdt = -beta * S * I / N
            dIdt = beta * S * I / N - gamma * I
            dRdt = gamma * I
            return dSdt, dIdt, dRdt

        # вектор начальных значений
        y0 = s0, i0, r0

        fig = plt.figure()
        ax = fig.add_subplot()

        if use_generated_population_nodes_degrees:
            if self.population_nodes_degrees is None:
                # получить id людей, которые работают на предприятиях
                ind = self.population_total.query("manufacture_number != -1")["id"]

                # создать граф
                tmp = self.connections_matrix[ind, :].tocsc()[:, ind].toarray()

                # исключить контакты людей самих с собой
                np.fill_diagonal(tmp, 0)
                graph = nx.DiGraph(tmp)
                self.population_nodes_degrees = nx.degree_histogram(graph)

            self.population_nodes_degrees = np.array(self.population_nodes_degrees)
            beta_inner = (self.population_nodes_degrees - self.population_nodes_degrees.min()) / \
                         (self.population_nodes_degrees.max() - self.population_nodes_degrees.min())
        else:
            beta_inner = beta

        for i, b in enumerate(beta_inner):
            ret = odeint(deriv, y0, t, args=(population_size, b, gamma))
            S, I, R = ret.T
            if plot_s:
                ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible' if i == 0 else None)
            if plot_i:
                ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected' if i == 0 else None)
            if plot_r:
                ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered' if i == 0 else None)

        ax.set_xlabel('Дни')
        ax.set_ylabel('Количество')
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid()
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)
        plt.show()

        return 0

    def plot_households_distribution(self) -> int:

        households_grouped = self.households_distribution_template[["region_type", "1_person", "2_persons",
                                                                    "3_persons", "4_persons", "5_persons",
                                                                    "6+_persons"]] \
                                 .groupby("region_type") \
                                 .sum() \
                                 .reset_index()

        plt.figure(figsize=(20, 10))
        fig, axs = plt.subplots(1, 2, figsize=(20, 8))

        axs[0].set_ylim(0, 4.5 * 10 ** 7)
        axs[1].set_ylim(0, 4.5 * 10 ** 7)

        df_melted_urban = households_grouped \
            .query("region_type == 'Городские населенные пункты'")[["1_person", "2_persons", "3_persons",
                                                                    "4_persons", "5_persons", "6+_persons"]] \
            .rename(columns={"1_person": "1 человек",
                             "2_persons": "2 человека",
                             "3_persons": "3 человека",
                             "4_persons": "4 человека",
                             "5_persons": "5 человек",
                             "6+_persons": "Более 6 человек"}) \
            .melt()

        df_melted_rural = households_grouped \
            .query("region_type == 'Сельские населенные пункты'")[["1_person", "2_persons", "3_persons",
                                                                   "4_persons", "5_persons", "6+_persons"]] \
            .rename(columns={"1_person": "1 человек",
                             "2_persons": "2 человека",
                             "3_persons": "3 человека",
                             "4_persons": "4 человека",
                             "5_persons": "5 человек",
                             "6+_persons": "Более 6 человек"}) \
            .melt()

        sns.barplot(x='variable', y='value', data=df_melted_rural, ax=axs[1])
        sns.barplot(x='variable', y='value', data=df_melted_urban, ax=axs[0])

        axs[0].set_title("Городское население")
        axs[0].set_xlabel("Число людей в семье")
        axs[0].set_ylabel("Количество семей")

        axs[1].set_title("Сельское население")
        axs[1].set_xlabel("Число людей в семье")
        axs[1].set_ylabel("Количество семей")

        return 0


    def generate_total_population(self,
                                  population_size: int) -> int:

        """ Метод создает популяцию городскую + сельскую """
        print(datetime.datetime.now(), ": Строится популяция для городского населения ... ")
        self.create_population(population_type="urban",
                               population_size=(population_size // 2),
                               largest_manufactures_number=12)

        urban_population_raw = self.population.copy()
        urban_population = self.population.groupby(["age_group", "sex", "population_type"],
                                                   as_index=False).count()

        print(datetime.datetime.now(), ": Строится популяция для сельского населения ... ")
        self.create_population(population_type="rural",
                               population_size=(population_size // 2),
                               largest_manufactures_number=12)

        rural_population_raw = self.population.copy()
        rural_population = self.population.groupby(["age_group", "sex", "population_type"],
                                                   as_index=False).count()

        self.population_total = pd.concat([urban_population_raw, rural_population_raw])

        population_predicted = urban_population.query("sex == 'man'")[["age_group", "id"]] \
            .rename(columns={"id": "men_urban"}) \
            .merge(urban_population.query("sex == 'woman'")[["age_group", "id"]] \
                   .rename(columns={"id": "women_urban"}),
                   on="age_group", how="left") \
            .merge(rural_population.query("sex == 'woman'")[["age_group", "id"]] \
                   .rename(columns={"id": "women_rural"}),
                   on="age_group", how="left") \
            .merge(rural_population.query("sex == 'man'")[["age_group", "id"]] \
                   .rename(columns={"id": "men_rural"}),
                   on="age_group", how="left")

        population_predicted = population_predicted.set_index("age_group")
        population_predicted = population_predicted.loc[
                               ['0 – 4', '5 – 9', '10 – 14', '15 – 19', '20 – 24', '25 – 29', '30 – 34',
                                '35 – 39', '40 – 44', '45 – 49', '50 – 54', '55 – 59',
                                '60 – 64', '65 – 69', '70 лет и более'], :].reset_index()

        self.population_total_grouped = population_predicted
        return 0


    def plot_households_distribution_generated_population(self,
                                                          population_size: int) -> int:
        """ Метод рисует распределение по домохозяйствам для сгенерированной популяции"""
        if self.population_total is None:
            self.generate_total_population(population_size=population_size)

        tmp = self.population_total.query("population_type == 'urban'").groupby("household_id").count().groupby(
            "id").count().reset_index()[["id", "age"]].head(6).transpose()
        tmp.columns = tmp.loc["id", :]
        tmp = tmp.tail(1)
        tmp_urban = tmp.melt()

        tmp = self.population_total.query("population_type == 'rural'").groupby("household_id").count().groupby(
            "id").count().reset_index()[["id", "age"]].head(6).transpose()
        tmp.columns = tmp.loc["id", :]
        tmp = tmp.tail(1)
        tmp_rural = tmp.melt()


        plt.figure(figsize=(20, 10))
        fig, axs = plt.subplots(1, 2, figsize=(20, 8))
        plt.style.use('ggplot')

        axs[0].set_ylim(0, 5000)
        axs[1].set_ylim(0, 5000)

        sns.barplot(x='id', y='value', data=tmp_rural, ax=axs[1])
        sns.barplot(x='id', y='value', data=tmp_urban, ax=axs[0])

        axs[0].set_title("Городское население")
        axs[0].set_xlabel("Число людей в семье")
        axs[0].set_ylabel("Количество семей")

        axs[1].set_title("Сельское население")
        axs[1].set_xlabel("Число людей в семье")
        axs[1].set_ylabel("Количество семей")

        return 0

    def plot_age_sex_distribution(self,
                                  max_xlimit: int,
                                  population_size: int,
                                  use_generated_population: bool = False,
                                  age_sex_distribution_filename: str = "age_sex_distribution_percentage.xlsx") -> int:
        """ Внимание: этот метод меняет атрибут population !!!"""

        # 1 Подготовка данных
        # различаются случаи, когда надо построить популяцию по фактическим данным и когда надо
        # сгенерировать популяцию на основе шаблона
        if not use_generated_population:

            print(datetime.datetime.now(), ": Выполняется отрисовка возрастных пирамид для фактических данных ... ")
            self.read_age_sex_distribution_template(file_name=age_sex_distribution_filename)
            age_sex_distribution_template_inner = self.age_sex_distribution_template.copy()

            # 1.2 сгруппировать данные по возрастным группам
            age_sex_distribution_template_inner = \
                age_sex_distribution_template_inner[["men_women_total", "men_total", "women_total", "men_women_urban",
                                                     "men_urban", "women_urban", "men_women_rural", "men_rural",
                                                     "women_rural", "age_group"]] \
                .groupby("age_group") \
                .sum() \
                .loc[['0 – 4', '5 – 9', '10 – 14', '15 – 19', '20 – 24', '25 – 29', '30 – 34',
                      '35 – 39', '40 – 44', '45 – 49', '50 – 54', '55 – 59',
                      '60 – 64', '65 – 69', '70 лет и более'], :] \
                .reset_index()

            # 1.3 рассчитать число людей каждой группы исходя из размера популяции
            age_sex_distribution_template_inner[["men_women_total", "men_total", "women_total", "men_women_urban",
                                                 "men_urban", "women_urban", "men_women_rural", "men_rural",
                                                 "women_rural"]] *= (population_size // 4)

            age_sex_distribution_template_inner \
                .loc[age_sex_distribution_template_inner.age_group == "70 лет и более", "age_group"] \
                    = "70 лет \n и более"
        else:
            print(datetime.datetime.now(),
                  ": Выполняется отрисовка возрастных пирамид для сгенерированной популяции ... ")

            if self.population_total_grouped is None:
                self.generate_total_population(population_size=population_size)

            age_sex_distribution_template_inner = self.population_total_grouped

        # 2. Отрисовка графика
        plt.figure(figsize=(20, 10))
        fig, axs = plt.subplots(1, 2, figsize=(22, 8))

        axs[0].set_xlim(-max_xlimit, max_xlimit)
        axs[1].set_xlim(-max_xlimit, max_xlimit)

        AgeClass = age_sex_distribution_template_inner["age_group"]

        sns.barplot(x=-1 * age_sex_distribution_template_inner.men_urban, y="age_group",
                    data=age_sex_distribution_template_inner,
                    order=AgeClass, color="red", label="мужчины", ax=axs[0])

        sns.barplot(x=age_sex_distribution_template_inner.women_urban, y="age_group",
                    data=age_sex_distribution_template_inner,
                    order=AgeClass, color="blue", label="женщины", ax=axs[0])

        sns.barplot(x=-1 * age_sex_distribution_template_inner.men_rural, y="age_group",
                    data=age_sex_distribution_template_inner,
                    order=AgeClass, color="red", label="мужчины", ax=axs[1])

        sns.barplot(x=age_sex_distribution_template_inner.women_rural, y="age_group",
                    data=age_sex_distribution_template_inner,
                    order=AgeClass, color="blue", label="женщины", ax=axs[1])

        axs[0].set_title("Городское население")
        axs[0].set_xlabel("Численность")
        axs[0].set_ylabel("Возрастная группа")

        axs[1].set_title("Сельское население")
        axs[1].set_xlabel("Численность")
        axs[1].set_ylabel("Возрастная группа")

        axs[0].axvline(0, 1, 0, color="black")
        axs[0].legend(prop={'size': 15})

        axs[1].axvline(0, 1, 0, color="black")
        axs[1].legend(prop={'size': 15})

        def abs_fmt(x, pos):
            """Функция форматирования для модуля"""
            return f'{int(abs(x))}'

        from matplotlib.ticker import FuncFormatter
        for ax in axs.flatten():
            ax.xaxis.set_major_formatter(FuncFormatter(abs_fmt))

        return 0



    def read_age_sex_distribution_template(self, file_name: str,
                                           input_folder: str = "") -> int:
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
        tmp = pd.concat([df_inner.iloc[:-1, :]] * 5).reset_index(drop=True)

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

        self.age_sex_distribution_template = tmp

        return 0

    def read_manufactures_distribution_template(self, file_name: str,
                                                input_folder: str = "") -> 0:
        """ Функция загружает шаблон распределения предприятий по регионам России """

        # загрузить шаблон, убрать пустые строки и столбцы
        df_inner = pd.read_excel(input_folder + file_name).iloc[5:, 1:]

        df_inner.columns = pd.Index(["Название округа ", "Название области", "Код ОКАТО", "Сельское, лесное хозяйство",
                                     "Добыча полезных ископаемых", "Обрабатывающие производства",
                                     "Обеспечение электрической энергией, газом и паром", "Водоснабжение",
                                     "Строительство",
                                     "Торговля оптовая и розничная", "Транспортировка и хранение",
                                     "Деятельность гостиниц и предприятий общественного питания",
                                     "Деятельность в области информации и связи", "Деятельность финансовая и страховая",
                                     "Деятельность по операциям с недвижимым имуществом",
                                     "Деятельность профессиональная, научная и техническая",
                                     "Государственное управление и обеспечение военной безопасности", "Образование",
                                     "Деятельность в области здравоохранения",
                                     "Деятельность в области культуры, спорта",
                                     "Предоставление прочих видов услуг",
                                     "Недифференцированная деятельность частных домашних хозяйств",
                                     "Деятельность экстерриториальных организаций и органов"])

        self.manufactures_distribution_template = df_inner
        return 0

    def read_schools_distribution_template(self, file_name: str,
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

        self.schools_distribution_template = df_inner
        return 0

    def _get_household_ratio(self) -> int:

        tmp = self.households_distribution_template[["region_type", "1_person", "2_persons",
                                                     "3_persons", "4_persons", "5_persons", "6+_persons"]] \
            .groupby("region_type").sum()

        urban = tmp.loc['Городские населенные пункты', :].to_numpy()
        rural = tmp.loc['Сельские населенные пункты', :].to_numpy()

        # перевести данные в доли
        urban = urban / urban.sum()
        rural = rural / rural.sum()

        self.household_ratio_urban = urban
        self.household_ratio_rural = rural
        return 0

    def _create_population_from_data(self, population_type: Literal["men", "women"],
                                     distribution_template: pd.DataFrame) -> pd.DataFrame:

        population = distribution_template[["age", population_type]]

        population = population.loc[np.repeat(population.index, population[population_type])] \
            .reset_index(drop=True) \
            .reset_index() \
            .drop(columns=[population_type])

        population["sex"] = "man" if population_type == "men" else "woman"
        population["household_id"] = np.nan
        population = population.rename(columns={"index": "id_in_sex_group"})
        population["system_record_number"] = population["id_in_sex_group"].astype("str") \
                                             + population["age"].astype("str") \
                                             + population["sex"].astype("str")

        population = population[["system_record_number", "age", "sex", "id_in_sex_group", "household_id"]]

        return population

    def _form_household(self, number_of_children: int,
                        population_inner: pd.DataFrame,
                        household_size: int,
                        number_of_parents: int = 2):

        def _find_young_index(a: np.ndarray) -> int:
            for i in range(a.size - 1):
                if (a[i] >= 18) and (a[i + 1] >= 18):
                    return i
            return -1

        _population_inner = population_inner.copy()

        elder_then_18_index = _find_young_index(population_inner["age"].to_numpy())

        household_parents = population_inner.loc[elder_then_18_index:, :] \
                                .iloc[:(household_size - 1) * number_of_parents
        if household_size % 2 == 1
        else household_size * number_of_parents, :]

        household_parents["household_id"] = np.arange(household_parents.shape[0] // number_of_parents) \
            .repeat(number_of_parents)

        if number_of_children:
            household_parents["role"] = "parent"

            # раньше детей выбирал так, но терялась одна возрастная категория, из-за того что есть дисбаланс
            # по полу в возрастных категориях
            # household_children = population_inner.loc[:elder_then_18_index, :] \
            #                                     .iloc[:(household_size - 1) * number_of_children
            #                                           if household_size % 2 == 1
            #                                           else household_size * number_of_children, :]
            # students = np.random.choice(ind, min(average_class_size, ind.size), replace=False)

            subset = np.random.choice(np.arange(elder_then_18_index),
                                      (household_size - 1) * number_of_children
                                      if household_size % 2 == 1
                                      else household_size * number_of_children,
                                      replace=False)

            household_children = population_inner.loc[:elder_then_18_index, :] \
                                     .iloc[subset, :]

            household_children["household_id"] = np.arange(household_children.shape[0] // number_of_children) \
                .repeat(number_of_children)

            household_children["role"] = "child"
            household = pd.concat([household_parents, household_children]).sort_values("household_id").reset_index(
                drop=True)
        else:
            household_parents["role"] = "no children"
            household = household_parents

        # удалить из основной популяции образовавшиеся семьи
        _population_inner = _population_inner[~_population_inner["system_record_number"] \
            .isin(household["system_record_number"])] \
            .reset_index(drop=True)

        return _population_inner, household

    def _create_connections_in_population(self,
                                          households_number: pd.Series) -> int:

        population_inner = self.population.sort_values(["id_in_sex_group", "sex"]).reset_index(drop=True)

        # 1.1 сформировать семьи из 2х человек
        population_inner, two_p = self._form_household(number_of_children=0,
                                                       population_inner=population_inner,
                                                       household_size=households_number["2_persons"],
                                                       number_of_parents=2)

        # 1.2 сформировать семьи из 3х человек
        population_inner, three_p = self._form_household(number_of_children=1,
                                                         population_inner=population_inner,
                                                         household_size=households_number["3_persons"],
                                                         number_of_parents=2)

        # 1.3 сформировать семьи из 4х человек
        population_inner, four_p = self._form_household(number_of_children=2,
                                                        population_inner=population_inner,
                                                        household_size=households_number["4_persons"],
                                                        number_of_parents=2)

        # 1.4 сформировать семьи из 5 человек
        population_inner, five_p = self._form_household(number_of_children=3,
                                                        population_inner=population_inner,
                                                        household_size=households_number["5_persons"],
                                                        number_of_parents=2)

        # 1.5 сформировать семьи из 6 человек
        population_inner, six_p = self._form_household(number_of_children=4,
                                                       population_inner=population_inner,
                                                       household_size=households_number["6+_persons"],
                                                       number_of_parents=2)

        # 1.5 сформировать семьи из 1 человека
        population_inner, one_p = self._form_household(number_of_children=0,
                                                       population_inner=population_inner,
                                                       household_size=households_number["1_person"],
                                                       number_of_parents=1)

        # 1.6 сделать номера домохозяйств уникальными
        two_p["household_id"] = two_p["household_id"] + one_p["household_id"].max()
        three_p["household_id"] = three_p["household_id"] + two_p["household_id"].max()
        four_p["household_id"] = four_p["household_id"] + three_p["household_id"].max()
        five_p["household_id"] = five_p["household_id"] + four_p["household_id"].max()
        six_p["household_id"] = six_p["household_id"] + five_p["household_id"].max()

        # 2 сформировать итоговую популяцию
        population = pd.concat([one_p, two_p, three_p, four_p, five_p, six_p]).reset_index(drop=True).reset_index() \
            .rename(columns={"index": "id"})

        self.population = population
        return 0

    def _create_contacts_inside_households(self,
                                           weight: int = 1) -> int:
        """ Функция создает матрицу контактов внутри домохозяйства """

        connections_matrix = lil_matrix((self.population.shape[0], self.population.shape[0]), dtype=np.int8)

        unique_vals = np.unique(self.population["household_id"])

        all_combinations = []
        for elem in unique_vals:
            indices = np.where(self.population["household_id"] == elem)[0].tolist()
            all_combinations += list(itertools.combinations_with_replacement(indices, 2))

        row_indices, col_indices = zip(*all_combinations)
        connections_matrix[row_indices, col_indices] = weight
        connections_matrix[col_indices, row_indices] = weight

        self.connections_matrix = connections_matrix
        return 0

    def _add_connections_to_matrix(self,
                                   nodes: np.ndarray,
                                   weight: int) -> int:
        """ Функция добавляет в матрицу контактов всевозможные комбинации из переданных индексов """

        all_combinations = list(itertools.combinations_with_replacement(nodes, 2))

        row_indices, col_indices = zip(*all_combinations)
        self.connections_matrix[row_indices, col_indices] = weight
        self.connections_matrix[col_indices, row_indices] = weight

        return 0

    def _create_connections_inside_schools(self,
                                           average_school_size: int,
                                           weight: int = 10) -> int:
        """ Функция создает связи внутри классов по школам """

        average_class_size = int(average_school_size // 11)

        number_of_schools = int(self.population.query("age <= 18").shape[0] * 0.75 / average_school_size)

        population_inner = self.population.copy()
        population_inner["school_number"] = -1

        # создать словарь с индексами по возрастным категориям
        for i in range(number_of_schools):
            for age in range(7, 18, 1):
                # выбрать детей, которые будут ходить в школу
                # из числа детей, подходящих по возрасту
                ind = population_inner.query("(age == @age) & (school_number == -1)")["id"]
                students = np.random.choice(ind, min(average_class_size, ind.size), replace=False)

                if not len(students):
                    continue

                # закрепить за этими детьми школу
                population_inner.loc[students, "school_number"] = i

                self._add_connections_to_matrix(nodes=students, weight=weight)

        return 0

    def plot_manufactures_size_distribution(self) -> Literal[0, 1]:
        """ Функция рисует, как распределены предприятия по размерам """

        if self.display_status:
            print(datetime.datetime.now(), ": Рисуется распределение домохозяйств по размеру ... ")

        fig, ax = plt.subplots()
        keys, values = list(self.manufactures_sizes_dict.keys()), list(self.manufactures_sizes_dict.values())

        # поставить логарифмическую шкалу
        ax.set_xscale('log')

        # построить график
        ax.plot(keys, values)
        ax.scatter(keys, values)
        ax.grid(True)
        ax.set_xlabel("Размер предприятия (число сотрудников)")
        ax.set_ylabel("Количество предприятий")
        ax.set_title("Распределение предприятий по размеру")
        plt.show()

        return 0

    def plot_manufactures_connections_graph(self, manufacture_id: int = -1) -> Literal[0, 1]:
        """ Функция рисует граф контактов внутри предприятия"""

        np.random.seed(self.random_seed)

        if manufacture_id == -1:
            manufacture_id_inner = np.random.choice(np.array(self.population["manufacture_number"] \
                                                                 [self.population["manufacture_number"] != -1]), 1)[0]
        else:
            manufacture_id_inner = manufacture_id
        print(manufacture_id_inner)
        # получить id людей, которые работают на этом предприятии
        ind = self.population.query("manufacture_number == @manufacture_id_inner")["id"]

        # создать граф
        tmp = self.connections_matrix[ind, :].tocsc()[:, ind].toarray()
        np.fill_diagonal(tmp, 0)
        graph = nx.DiGraph(tmp)
        pos = nx.circular_layout(graph)

        nx.draw(G=graph, pos=pos)
        plt.show()

        return 0

    def plot_total_connections_hist(self,
                                    bins: int = 15) -> Literal[0, 1]:
        """ Нарисовать гистограмму распределения степеней вершин для всего графа"""

        # получить id людей, которые работают на предприятиях
        ind = self.population_total.query("manufacture_number != -1")["id"]

        # создать граф
        tmp = self.connections_matrix[ind, :].tocsc()[:, ind].toarray()

        # исключить контакты людей самих с собой
        np.fill_diagonal(tmp, 0)

        print(datetime.datetime.now(), ": Вычисление степеней вершин для матрицы контактов ... ")
        graph = nx.DiGraph(tmp)
        self.population_nodes_degrees = nx.degree_histogram(graph)

        # нарисовать гистограмму
        x = np.repeat(range(len(self.population_nodes_degrees)), self.population_nodes_degrees)
        plt.hist(x, bins=bins)
        plt.xlabel('Степень вершины')
        plt.ylabel('Число вершин')
        plt.title('Гистограмма степеней вершин популяции')
        plt.show()

        return 0

    def plot_manufactures_connections_hist(self, manufacture_id: int = -1) -> Literal[0, 1]:
        """ Функция рисует граф степеней вершин графа контактов внутри предприятия"""

        np.random.seed(self.random_seed)
        # выбрать номер предприятия, но не учитывать людей без предприятия (номер -1)
        if manufacture_id == -1:
            manufacture_id_inner = np.random.choice(np.array(self.population["manufacture_number"] \
                                                            [self.population["manufacture_number"] != -1]), 1)[0]
        else:
            manufacture_id_inner = manufacture_id

        # получить id людей, которые работают на этом предприятии
        ind = self.population.query("manufacture_number == @manufacture_id_inner")["id"]

        # создать граф
        tmp = self.connections_matrix[ind, :].tocsc()[:, ind].toarray()
        np.fill_diagonal(tmp, 0)
        graph = nx.DiGraph(tmp)

        hist = nx.degree_histogram(graph)

        # Рисуем гистограмму
        plt.bar(range(len(hist)), hist, width=0.8, color='b')
        plt.xlabel('Степень вершины')
        plt.ylabel('Число вершин')
        plt.title('Гистограмма степеней вершин для предприятия ' + str(manufacture_id_inner))
        plt.show()

        return 0

    def _create_connections_inside_manufactures(self,
                                                largest_manufactures_number: int,
                                                weight: int = 20,
                                                betta: float = 0.3) -> int:

        # определить число людей работоспособного возраста
        number_of_workers = self.population.query("age > 18").shape[0]

        self.population["manufacture_number"] = -1

        # определить порядок числа number_of_workers
        order = int(np.floor(np.log10(number_of_workers)))

        a = np.power(10, np.arange(order))[::-1]

        m = int((number_of_workers - a @ np.ones(a.size) * largest_manufactures_number) // (a @ np.arange(a.size)))
        if m <= 0:
            print(m)
            raise Exception

        self.manufactures_sizes_series = pd.Series(
            np.repeat(a, m * np.arange(a.size) + largest_manufactures_number)).value_counts()
        self.manufactures_sizes_dict = self.manufactures_sizes_series.to_dict()

        for manufacture_size in self.manufactures_sizes_dict.keys():
            for j in range(self.manufactures_sizes_dict[manufacture_size]):

                # выбрать людей трудоспособного возраста, за которыми не закреплено предприятие
                ind = self.population.query("(age >= 19) & (manufacture_number == -1)")["id"]

                # выбрать людей, которых закрепим за текущим предприятием
                workers = np.random.choice(ind, manufacture_size, replace=False)
                if not len(workers):
                    continue

                # закрепить за этими людьми номер предприятие
                self.population.loc[workers, "manufacture_number"] = j * manufacture_size

                # создать связи между людьми по схеме small-world
                number_of_nodes = len(workers)
                # сделать так, чтобы изначально все workers стоят по кругу и соединены только с соседом
                for i in range(len(workers) - 1):
                    self.connections_matrix[workers[i], workers[i + 1]] = 1
                    self.connections_matrix[workers[i + 1], workers[i]] = 1
                self.connections_matrix[workers[0], workers[-1]] = 1
                self.connections_matrix[workers[-1], workers[0]] = 1

                number_of_ripped_edges = int(number_of_nodes * betta)
                index = np.random.choice(workers, number_of_ripped_edges, replace=False)
                for i in index:
                    if i != workers.size - 1:
                        # разорвать связь
                        self.connections_matrix[i, i + 1] = 0
                        self.connections_matrix[i + 1, i] = 0
                        # добавить новую случайную связь
                        new_node = np.random.choice(workers[workers != i], 1, replace=False)[0]
                        self.connections_matrix[i, new_node] = 1
                        self.connections_matrix[new_node, i] = 1
                    else:
                        # разорвать связь
                        self.connections_matrix[0, 0 + 1] = 0
                        self.connections_matrix[0 + 1, 0] = 0
                        # добавить новую случайную связь
                        new_node = np.random.choice(workers[workers != 0], 1, replace=False)[0]
                        self.connections_matrix[0, new_node] = 1
                        self.connections_matrix[new_node, 0] = 1

        return 0

    def _create_random_connections(self,
                                   power: float,
                                   weight: int = 40) -> int:

        workers = np.random.choice(self.population["id"], int(self.population["id"].shape[0] * power), replace=False)
        self._add_connections_to_matrix(nodes=workers, weight=weight)

        return 0

    def plot_graph(self,
                   display_status: bool = True) -> int:
        """ Функция рисует граф по матрице контактов """

        # plt.figure(facecolor='beige')
        plt.rcParams['axes.facecolor'] = 'black'

        if display_status:
            print(datetime.datetime.now(), ": Создается граф по матрице контактов ... ")

        matrix_inner = self.connections_matrix.toarray()
        np.fill_diagonal(matrix_inner, 0)

        color_map = {10: 'r', 20: 'g', 30: 'black', 40: 'y', 50: 'blue'}

        graph = nx.DiGraph(matrix_inner)

        if display_status:
            print(datetime.datetime.now(), ": Выполняется отрисовка построенного графа ... ")

        nx.draw(G=graph, node_size=10, arrows=False, with_labels=False, width=2.0, alpha=0.2)

        # Создание легенды
        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor='r', markersize=10, label='Домохозяйства'),
                            plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor='g', markersize=10, label='Школы'),
                            plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor='black', markersize=10, label='Предприятия'),
                            plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor='y', markersize=10, label='Университеты'),
                            plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor='blue', markersize=10, label='Случайные')])

        plt.show()

        return 0

    def plot_minimum_spanning_tree(self,
                                   display_status: bool = True) -> int:
        """ Функция рисует остовное дерево по матрце контактов """

        if display_status:
            print(datetime.datetime.now(), ": Создается граф по матрице контактов ... ")

        matrix_inner = self.connections_matrix.toarray()
        np.fill_diagonal(matrix_inner, 0)

        mst = nx.minimum_spanning_tree(nx.from_numpy_array(matrix_inner))

        if display_status:
            print(datetime.datetime.now(), ": Выполняется отрисовка построенного графа ... ")

        nx.draw(mst, node_color='blue', node_size=10, edge_color='grey')
        plt.title('Остовное дерево')

        plt.show()
        return 0

    def plot_heat_map(self) -> int:
        """ Функция рисует тепловую карту матрицы контактов """
        plt.imshow(self.connections_matrix.toarray(), cmap="hot", interpolation="nearest")
        plt.colorbar()
        plt.show()
        return 0

    def _create_connection_inside_university(self,
                                             average_university_size: int,
                                             average_number_of_groups: int,
                                             weight: int = 30) -> int:

        average_group_size = int(average_university_size // average_number_of_groups)
        number_of_universities = int(
            self.population.query("(age > 18) & (age < 27)").shape[0] / average_university_size)

        self.population["university_number"] = -1

        # создать словарь с индексами по возрастным категориям
        for i in tqdm(range(number_of_universities)):
            for department in range(average_number_of_groups):

                ind = self.population.query("(age > 18) & (age < 27) & (university_number == -1)")["id"]
                students = np.random.choice(ind, min(average_group_size, ind.size), replace=False)

                if not len(students):
                    continue

                # закрепить за этими детьми школу
                self.population.loc[students, "university_number"] = i

                self._add_connections_to_matrix(nodes=students,
                                                weight=weight)

        return 0

    def create_population(self,
                          population_type: Literal["urban", "rural"],
                          population_size: int,
                          largest_manufactures_number: int = 20,
                          households_filename: str = "households.xlsx",
                          age_sex_distribution_filename: str = "age_sex_distribution_percentage.xlsx",
                          manufactures_filename: str = "manufactures.xlsx",
                          schools_filename: str = "schools.xlsx") -> int:

        print(datetime.datetime.now(), ": Запуск функции создания популяции ... ")

        self.read_households_distribution_template(file_name=households_filename)
        self.read_age_sex_distribution_template(file_name=age_sex_distribution_filename)
        self.read_manufactures_distribution_template(file_name=manufactures_filename)
        self.read_schools_distribution_template(file_name=schools_filename)

        # посчитать, сколько человек надо на каждый тип домохозяйства
        self._get_household_ratio()

        households_number = np.zeros_like(self.household_ratio_urban)
        # найти число людей для каждого типа домохозяйства
        if population_type == "urban":
            _population_type = self.household_ratio_urban
        elif population_type == "rural":
            _population_type = self.household_ratio_rural
        else:
            raise Exception

        households_number = _population_type * population_size // np.arange(1, 7, 1)

        # 1.1 найти число людей для каждой возрастной группы
        base = self.age_sex_distribution_template[["age", "men" + "_" + population_type,
                                                   "women" + "_" + population_type]] \
            .rename(
            columns={"men" + "_" + population_type: "men",
                     "women" + "_" + population_type: "women"}
        )

        # 1.2 считаем что мужчин и женщин равное количество в популяции
        # определяем, сколько человек в каждой возрастной категории
        base.loc[:, ["men", "women"]] = base.loc[:, ["men", "women"]] * population_size * 0.5
        base.loc[base["age"] == '70 лет и более', "age"] = "70"
        base = base.astype({'men': int, 'women': int, "age": int})

        print(datetime.datetime.now(), ": Создается популяция мужчин на основе шаблона ... ")
        men = self._create_population_from_data(population_type="men", distribution_template=base)

        print(datetime.datetime.now(), ": Создается популяция женщин на основе шаблона ... ")
        women = self._create_population_from_data(population_type="women", distribution_template=base)

        # 2 создать популяции из мужчин и женщин согласно шаблону
        self.population = pd.concat([men, women]).reset_index(drop=True)

        self.population["age_group"] = np.select([(self.population.age >= 0) & (self.population.age <= 4),
                                                  (self.population.age >= 5) & (self.population.age <= 9),
                                                  (self.population.age >= 10) & (self.population.age <= 14),
                                                  (self.population.age >= 15) & (self.population.age <= 19),
                                                  (self.population.age >= 20) & (self.population.age <= 24),
                                                  (self.population.age >= 25) & (self.population.age <= 29),
                                                  (self.population.age >= 30) & (self.population.age <= 34),
                                                  (self.population.age >= 35) & (self.population.age <= 39),
                                                  (self.population.age >= 40) & (self.population.age <= 44),
                                                  (self.population.age >= 45) & (self.population.age <= 49),
                                                  (self.population.age >= 50) & (self.population.age <= 54),
                                                  (self.population.age >= 55) & (self.population.age <= 59),
                                                  (self.population.age >= 60) & (self.population.age <= 64),
                                                  (self.population.age >= 65) & (self.population.age <= 69),
                                                  (self.population.age >= 70)],
                                                 ['0 – 4', '5 – 9', '10 – 14', '15 – 19', '20 – 24', '25 – 29',
                                                  '30 – 34',
                                                  '35 – 39', '40 – 44', '45 – 49', '50 – 54', '55 – 59', '60 – 64',
                                                  '65 – 69',
                                                  '70 лет и более'])

        # 2.1 разбить полученную популяцию на домохозяйства
        print(datetime.datetime.now(), ": Формирование домохозяйств ... ")
        self._create_connections_in_population(households_number=pd.Series(data=households_number.astype("int"),
                                                                           index=['1_person', '2_persons',
                                                                                  '3_persons', '4_persons',
                                                                                  '5_persons', '6+_persons']))

        # 3 создать матрицу контактов для внутри домохозяйств
        print(datetime.datetime.now(), ": Создание матрицы контактов внутри домохозяйств ... ")
        self._create_contacts_inside_households(weight=10)

        self.population["population_type"] = population_type

        # 4 создать матрицу контактов внутри школ
        print(datetime.datetime.now(), ": Создается контакты внутри школ ... ")
        average_school_size = int(self.schools_distribution_template["Число обучающихся на одну школу"].mean())

        self._create_connections_inside_schools(average_school_size=average_school_size,
                                                weight=10)

        # создать матрицу контактов внутри предприятий
        print(datetime.datetime.now(), ": Создается контакты внутри предприятий ... ")
        self._create_connections_inside_manufactures(largest_manufactures_number=largest_manufactures_number,
                                                     weight=10)

        # создать связи внутри университетов
        print(datetime.datetime.now(), ": Создается контакты внутри университетов ... ")
        self._create_connection_inside_university(average_university_size=300,
                                                  average_number_of_groups=30,
                                                  weight=10)

        # добавить случайные связи
        #print(datetime.datetime.now(), ": Создается слуяайные контакты внутри популяции ... ")
        #self._create_random_connections(power=0.01, weight=10)

        print(datetime.datetime.now(), ": Создание популяции завершено ... ")
        # connections_matrix = connections_matrix.toarray()
        # print(np.where(connections_matrix != np.transpose(connections_matrix)))
        # print(connections_matrix.toarray().shape)
        # print("---")
        # print(np.transpose(connections_matrix)[0])

        # plot_minimum_spanning_tree(connections_matrix)

        # plot_heat_map(connections_matrix)
        # plot_graph(connections_matrix)

        return 0
