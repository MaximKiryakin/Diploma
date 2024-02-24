import numpy as np
import pandas as pd # type: ignore
from typing import Tuple, Literal
from scipy.sparse import lil_matrix
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import datetime
from tqdm import tqdm


def _get_household_ratio(households_distribution_template: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    
    tmp = households_distribution_template[["region_type", "1_person", "2_persons",
                                            "3_persons",	"4_persons", "5_persons", "6+_persons"]] \
            .groupby("region_type").sum()
    
    urban = tmp.loc['Городские населенные пункты', :].to_numpy()
    rural = tmp.loc['Сельские населенные пункты', :].to_numpy()

    # перевести данные в доли 
    urban = urban / urban.sum()
    rural = rural / rural.sum()
    
    return urban, rural


def _create_population_from_data(population_type: Literal["men", "women"],
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


def _find_young_index(a: np.ndarray) -> int:
    for i in range(a.size - 1):
        if (a[i] >= 18) and (a[i + 1] >= 18):
            return i
    return -1


def _form_household(number_of_children: int,
                    population_inner: pd.DataFrame,
                    household_size: int,
                    number_of_parents: int = 2):

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
        #household_children = population_inner.loc[:elder_then_18_index, :] \
        #                                     .iloc[:(household_size - 1) * number_of_children
        #                                           if household_size % 2 == 1
        #                                           else household_size * number_of_children, :]
        #students = np.random.choice(ind, min(average_class_size, ind.size), replace=False)

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
        household = pd.concat([household_parents, household_children]).sort_values("household_id").reset_index(drop=True)
    else:
        household_parents["role"] = "no children"
        household = household_parents

    # удалить из основной популяции образовавшиеся семьи
    _population_inner = _population_inner[~_population_inner["system_record_number"]\
                                          .isin(household["system_record_number"])]\
                        .reset_index(drop=True)

    return _population_inner, household


def _create_connections_in_population(population: pd.DataFrame,
                                      households_number: pd.Series) -> pd.DataFrame:

    population_inner = population.sort_values(["id_in_sex_group", "sex"]).reset_index(drop=True)

    # 1.1 сформировать семьи из 2х человек
    population_inner, two_p = _form_household(number_of_children=0,
                                              population_inner=population_inner,
                                              household_size=households_number["2_persons"],
                                              number_of_parents=2)

    # 1.2 сформировать семьи из 3х человек
    population_inner, three_p = _form_household(number_of_children=1,
                                                population_inner=population_inner,
                                                household_size=households_number["3_persons"],
                                                number_of_parents=2)

    # 1.3 сформировать семьи из 4х человек
    population_inner, four_p = _form_household(number_of_children=2,
                                               population_inner=population_inner,
                                               household_size=households_number["4_persons"],
                                               number_of_parents=2)

    # 1.4 сформировать семьи из 5 человек
    population_inner, five_p = _form_household(number_of_children=3,
                                               population_inner=population_inner,
                                               household_size=households_number["5_persons"],
                                               number_of_parents=2)

    # 1.5 сформировать семьи из 6 человек
    population_inner, six_p = _form_household(number_of_children=4,
                                              population_inner=population_inner,
                                              household_size=households_number["6+_persons"],
                                              number_of_parents=2)

    # 1.5 сформировать семьи из 1 человека
    population_inner, one_p = _form_household(number_of_children=0,
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
    population = pd.concat([one_p, two_p, three_p, four_p, five_p, six_p]).reset_index(drop=True).reset_index()\
                   .rename(columns={"index": "id"})

    return population


def _create_contacts_inside_households(population: pd.DataFrame,
                                       weight: int = 1) -> lil_matrix:
    """ Функция создает матрицу контактов внутри домохозяйства """

    connections_matrix = lil_matrix((population.shape[0], population.shape[0]), dtype=np.int8)

    unique_vals = np.unique(population["household_id"])

    all_combinations = []
    for elem in unique_vals:
        indices = np.where(population["household_id"] == elem)[0].tolist()
        all_combinations += list(itertools.combinations_with_replacement(indices, 2))

    row_indices, col_indices = zip(*all_combinations)
    connections_matrix[row_indices, col_indices] = weight
    connections_matrix[col_indices, row_indices] = weight

    return connections_matrix


def _add_connections_to_matrix(matrix: lil_matrix,
                               nodes: np.ndarray,
                               weight: int) -> lil_matrix:
    """ Функция добавляет в матрицу контактов всевозможные комбинации из переданных индексов """

    matrix_inner = matrix.copy()
    all_combinations = list(itertools.combinations_with_replacement(nodes, 2))

    row_indices, col_indices = zip(*all_combinations)
    matrix_inner[row_indices, col_indices] = weight
    matrix_inner[col_indices, row_indices] = weight

    return matrix_inner


def _create_connections_inside_schools(population: pd.DataFrame,
                                       matrix: lil_matrix,
                                       average_school_size: int,
                                       weight: int = 10) -> Tuple[pd.DataFrame, lil_matrix]:
    """ Функция создает связи внутри классов по школам """

    average_class_size = int(average_school_size // 11)

    number_of_schools = int(population.query("age <= 18").shape[0]*0.75 / average_school_size)

    population_inner = population.copy()
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

            matrix = _add_connections_to_matrix(matrix=matrix, nodes=students, weight=weight)

    return population_inner, matrix


def _create_connections_inside_manufactures(population: pd.DataFrame,
                                            matrix: lil_matrix,
                                            average_manufacture_size: int,
                                            average_number_of_departments: int,
                                            weight: int = 20) -> Tuple[pd.DataFrame, lil_matrix]:

    average_department_size = int(average_manufacture_size // average_number_of_departments)
    number_of_manufactures = int(population.query("age > 18").shape[0] / average_manufacture_size)

    population_inner = population.copy()
    population_inner["manufacture_number"] = -1
    population_inner["department_number"] = -1

    # создать словарь с индексами по возрастным категориям
    for i in tqdm(range(number_of_manufactures)):
        for department in range(average_number_of_departments):

            up_border, down_border = 25 + department * 5, 19 + department * 5
            ind = population_inner.query("(age >= 19) & (age <= @up_border)"
                                         " & (age >= @down_border) & (manufacture_number == -1)")["id"]
            workers = np.random.choice(ind, min(average_department_size, ind.size), replace=False)

            if not len(workers):
                continue

            # закрепить за этими людьми номер предприятие
            population_inner.loc[workers, "manufacture_number"] = i

            # закрепить за этими людьми номер департамента в предприятии
            population_inner.loc[workers, "department_number"] = department

            # добавить связи внутри отдела
            matrix = _add_connections_to_matrix(matrix=matrix, nodes=workers, weight=weight)

            # добавить связи с соседними отделами
            #left_id = population_inner.query("department_number == @department - 1")["id"]
            #if left_id.size:
            #    ids = np.random.choice([0, 1], size=left_id.size, replace=True, p=[0.5, 0.5])
            #    matrix = _add_connections_to_matrix(matrix=matrix, nodes=np.concatenate((workers, ids)), weight=weight)



    return population_inner, matrix


def _create_random_connections(population: pd.DataFrame,
                               matrix: lil_matrix,
                               power: float,
                               weight: int = 40) -> lil_matrix:

    workers = np.random.choice(population["id"], int(population["id"].shape[0] * power), replace=False)
    matrix = _add_connections_to_matrix(matrix=matrix, nodes=workers, weight=weight)

    return matrix


def plot_graph(matrix: lil_matrix,
               display_status: bool = True) -> int:
    """ Функция рисует граф по матрице контактов """

    #plt.figure(facecolor='beige')
    plt.rcParams['axes.facecolor'] = 'black'

    if display_status:
        print(datetime.datetime.now(), ": Создается граф по матрице контактов ... ")

    matrix_inner = matrix.toarray()
    np.fill_diagonal(matrix_inner, 0)

    color_map = {10: 'r', 20: 'g', 30: 'black', 40: 'y', 50: 'blue'}

    graph = nx.DiGraph(matrix_inner)

    if display_status:
        print(datetime.datetime.now(), ": Выполняется отрисовка построенного графа ... ")

    nx.draw(G=graph, node_size=10, arrows=False, with_labels=False, width=2.0, alpha=0.2,
            edge_color=[color_map[graph[u][v]['weight']] for u, v in graph.edges()])

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


def plot_minimum_spanning_tree(matrix: lil_matrix,
                               display_status: bool = True) -> int:
    """ Функция рисует остовное дерево по матрце контактов """

    if display_status:
        print(datetime.datetime.now(), ": Создается граф по матрице контактов ... ")

    matrix_inner = matrix.toarray()
    np.fill_diagonal(matrix_inner, 0)

    mst = nx.minimum_spanning_tree(nx.from_numpy_array(matrix_inner))

    if display_status:
        print(datetime.datetime.now(), ": Выполняется отрисовка построенного графа ... ")

    nx.draw(mst, node_color='blue', node_size=10, edge_color='grey')
    plt.title('Остовное дерево')

    plt.show()
    return 0


def plot_heat_map(matrix: lil_matrix) -> int:
    """ Функция рисует тепловую карту матрицы контактов """
    plt.imshow(matrix.toarray())
    plt.show()
    return 0


def _create_connection_inside_university(population: pd.DataFrame,
                                         matrix: lil_matrix,
                                         average_university_size: int,
                                         average_number_of_groups: int,
                                         weight: int = 30) -> Tuple[pd.DataFrame, lil_matrix]:

    average_group_size = int(average_university_size // average_number_of_groups)
    number_of_universities = int(population.query("(age > 18) & (age < 27)").shape[0] / average_university_size)

    population_inner = population.copy()
    population_inner["university_number"] = -1

    # создать словарь с индексами по возрастным категориям
    for i in tqdm(range(number_of_universities)):
        for department in range(average_number_of_groups):

            ind = population_inner.query("(age > 18) & (age < 27) & (university_number == -1)")["id"]
            students = np.random.choice(ind, min(average_group_size, ind.size), replace=False)

            if not len(students):
                continue

            # закрепить за этими детьми школу
            population_inner.loc[students, "university_number"] = i

            matrix = _add_connections_to_matrix(matrix=matrix,
                                                nodes=students,
                                                weight=weight)

    return population_inner, matrix


def create_population(households_distribution_template: pd.DataFrame,
                      age_sex_distribution_template: pd.DataFrame,
                      population_type: Literal["urban", "rural"],
                      population_size: int,
                      schools_distribution_template: pd.DataFrame = None) -> Tuple[lil_matrix, pd.DataFrame]:

    print(datetime.datetime.now(), ": Запуск функции создания популяции ... ")
    # посчитать, сколько человек надо на каждый тип домохозяйства
    urban, rural = _get_household_ratio(households_distribution_template)

    households_number = np.zeros_like(urban)
    # найти число людей для каждого типа домохозяйства
    if population_type == "urban":
        _population_type = urban
    elif population_type == "rural":
        _population_type = rural
    else: 
        raise Exception

    households_number = _population_type * population_size // np.arange(1, 7, 1)

    # 1.1 найти число людей для каждой возрастной группы
    base = age_sex_distribution_template[["age", "men" + "_" + population_type,
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
    men = _create_population_from_data(population_type="men", distribution_template=base)

    print(datetime.datetime.now(), ": Создается популяция женщин на основе шаблона ... ")
    women = _create_population_from_data(population_type="women", distribution_template=base)

    # 2 создать популяции из мужчин и женщин согласно шаблону
    population = pd.concat([men, women]).reset_index(drop=True)

    population["age_group"] = np.select([(population.age >= 0) & (population.age <= 4),
                                         (population.age >= 5) & (population.age <= 9),
                                         (population.age >= 10) & (population.age <= 14),
                                         (population.age >= 15) & (population.age <= 19),
                                         (population.age >= 20) & (population.age <= 24),
                                         (population.age >= 25) & (population.age <= 29),
                                         (population.age >= 30) & (population.age <= 34),
                                         (population.age >= 35) & (population.age <= 39),
                                         (population.age >= 40) & (population.age <= 44),
                                         (population.age >= 45) & (population.age <= 49),
                                         (population.age >= 50) & (population.age <= 54),
                                         (population.age >= 55) & (population.age <= 59),
                                         (population.age >= 60) & (population.age <= 64),
                                         (population.age >= 65) & (population.age <= 69),
                                         (population.age >= 70)],
                                ['0 – 4', '5 – 9', '10 – 14', '15 – 19', '20 – 24', '25 – 29', '30 – 34',
                                         '35 – 39', '40 – 44', '45 – 49', '50 – 54', '55 – 59', '60 – 64', '65 – 69',
                                         '70 лет и более'])

    # 2.1 разбить полученную популяцию на домохозяйства
    print(datetime.datetime.now(), ": Формирование домохозяйств ... ")
    population = _create_connections_in_population(population=population,
                                                   households_number=pd.Series(data=households_number.astype("int"),
                                                                               index=['1_person', '2_persons',
                                                                                      '3_persons', '4_persons',
                                                                                      '5_persons', '6+_persons']))

    # 3 создать матрицу контактов для внутри домохозяйств
    print(datetime.datetime.now(), ": Создание матрицы контактов внутри домохозяйств ... ")
    connections_matrix = _create_contacts_inside_households(population=population,
                                                            weight=10)

    population["population_type"] = population_type

    # 4 создать матрицу контактов внутри школ
    print(datetime.datetime.now(), ": Создается контакты внутри школ ... ")
    average_school_size = int(schools_distribution_template["Число обучающихся на одну школу"].mean())

    population, connections_matrix = _create_connections_inside_schools(population=population,
                                                                        matrix=connections_matrix,
                                                                        average_school_size=average_school_size,
                                                                        weight=20)

    # создать матрицу контактов внутри предприятий
    print(datetime.datetime.now(), ": Создается контакты внутри предприятий ... ")
    population, connections_matrix = _create_connections_inside_manufactures(population=population,
                                                                             matrix=connections_matrix,
                                                                             average_manufacture_size=500,
                                                                             average_number_of_departments=10,
                                                                             weight=30)

    # создать связи внутри университетов
    print(datetime.datetime.now(), ": Создается контакты внутри университетов ... ")
    population, connections_matrix = _create_connection_inside_university(population=population,
                                                                          matrix=connections_matrix,
                                                                          average_university_size=300,
                                                                          average_number_of_groups=30,
                                                                          weight=40)

    # добавить случайные связи
    print(datetime.datetime.now(), ": Создается слуяайные контакты внутри популяции ... ")
    connections_matrix = _create_random_connections(population=population, matrix=connections_matrix, power=0.01,
                                                    weight=50)

    print(datetime.datetime.now(), ": Расчет окончен ... ")
    #connections_matrix = connections_matrix.toarray()
    #print(np.where(connections_matrix != np.transpose(connections_matrix)))
    #print(connections_matrix.toarray().shape)
    #print("---")
    #print(np.transpose(connections_matrix)[0])


    #plot_minimum_spanning_tree(connections_matrix)

    #plot_heat_map(connections_matrix)
    # plot_graph(connections_matrix)

    return connections_matrix, population
    