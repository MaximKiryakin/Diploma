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
#from tqdm import tqdm
from dataclasses import dataclass
from scipy.integrate import odeint
import matplotlib.cm as cm
from matplotlib import colors
from scipy import integrate




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
        self.connections_matrix_urban = None
        self.connections_matrix_rural = None
        self.connections_matrix = None
        self.population_nodes_degrees_urban = None
        self.population_nodes_degrees_rural = None
        self.population_nodes_degrees_absolute_values = None
        self.betta_dict = None
        self.mean_weight_beta = None
        self.moment_of_change_dict = None
        self.beta_inner= None
        self.solution = None
        self.beta_normalized = None
        self.time_linspace = None
        self.sir_model_population_type = None
        self.time_start = None
        self.time_end = None

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

    def calc_sir_model(self,
                       population_type: Literal["urban", "rural"],
                       gamma: float,
                       beta0: float,
                       t_end: int,
                       n_steps: int,
                       t_start: int = 0,
                       one_group_mode: bool = False) -> int:
        """ Метод решает систему дифференциальных уравнений для SIR модели """

        # 0. запомнить, для какого типа популяции делаем расчет
        self.sir_model_population_type = population_type

        # 1. вычислить значения бета и нормировать из в интервал [0, 1]
        nd = self.population_nodes_degrees_urban if population_type == "urban" else self.population_nodes_degrees_rural
        tmp = np.arange(nd.size)

        # 1.1 если задан режим без учета групп, положить число групп равным среднему по нормированным бета
        self.beta_normalized = (tmp - tmp.min()) / (tmp.max() - tmp.min())
        if one_group_mode:
            self.beta_normalized = np.array([self.beta_normalized.mean()])

        # 2. вычислить число компартментов
        num_age_groups = self.beta_normalized.size

        # 3. инициализировать число инфицированных в каждой группе единицами
        initial_infected = np.ones(num_age_groups)

        # 4. задать начальные условия (S0, I0, R0) для каждого компартмента
        initial_conditions = np.ones(3 * num_age_groups)

        for i in range(num_age_groups):
            group_size = nd[i] if not one_group_mode else nd.sum()
            initial_conditions[i * 3] = group_size - initial_infected[i]   # Susceptible
            initial_conditions[i * 3 + 1] = initial_infected[i]            # Infected
            initial_conditions[i * 3 + 2] = 0                              # Recovered

        def _sir_model(t, y, beta, gamma, num_age_groups, beta0):
            dydt = np.zeros_like(y)

            for i in range(num_age_groups):
                S = y[i * 3]
                I = y[i * 3 + 1]
                R = y[i * 3 + 2]

                """
                dSdt = -beta0 * S * (i + 1) * np.sum(
                    [(i + 1) * beta[i] * y[i * 3 + 1] for i in range(num_age_groups)]) // np.sum(
                    [(i + 1) * beta[i] for i in range(num_age_groups)])
                dIdt = beta0 * S * (i + 1) * np.sum(
                    [(i + 1) * beta[i] * y[i * 3 + 1] for i in range(num_age_groups)]) // np.sum(
                    [(i + 1) * beta[i] for i in range(num_age_groups)]) - gamma * I
                dRdt = gamma * I
                """
                dSdt = -beta0 * S * np.sum(
                    [beta[i] * y[i * 3 + 1] for i in range(num_age_groups)]) // np.sum(
                    [beta[i] for i in range(num_age_groups)])
                dIdt = beta0 * S * np.sum(
                    [beta[i] * y[i * 3 + 1] for i in range(num_age_groups)]) // np.sum(
                    [beta[i] for i in range(num_age_groups)]) - gamma * I
                dRdt = gamma * I

                dydt[i * 3] = dSdt
                dydt[i * 3 + 1] = dIdt
                dydt[i * 3 + 2] = dRdt

            return dydt

        # 5. создать временную сетку на которой решается дифференциальное уравнение
        self.time_linspace = np.linspace(t_start, t_end, n_steps)
        self.time_start = t_start
        self.time_end = t_end

        # 6. численно решить дифференциальное уравнение
        self.solution = integrate.solve_ivp(_sir_model, [self.time_linspace[0], self.time_linspace[-1]],
                                            initial_conditions, t_eval=self.time_linspace,
                                            args=(self.beta_normalized, gamma, num_age_groups, beta0)).y.T

        return 0

    def plot_sir_model(self,
                       plot_s: bool = False,         # рисовать ли кривую для восприимчивых
                       plot_i: bool = False,         # рисовать ли кривую для инфицированных
                       plot_r: bool = False,         # рисовать ли кривую для выздоровевших
                       title: str = "",              # заголовок графика
                       xlabel: str = "",             # подпись по оси х
                       ylabel: str = "",             # подпись по оси y
                       save_path: str = "",          # путь, по которому сохранить график
                       ylim: int = -1) -> int:

        """ Метод рисует график для SIR модели, по заранее посчитанному вектору-решению """

        # 1. вычислить число компартментов
        num_age_groups = self.beta_normalized.size

        # 2. нарисовать графики
        plt.figure(figsize=(10, 10))
        fig, axs = plt.subplots(figsize=(10, 10))

        # 2.1 задать шкалы для colorbar по номеру группы
        norm = colors.Normalize(vmin=0,
                                vmax=num_age_groups - 1 if num_age_groups > 1 else num_age_groups + 2)

        cmap = plt.get_cmap('tab20', num_age_groups - 1 if num_age_groups > 1 else num_age_groups + 2 )

        if self.sir_model_population_type == "urban":
            tmp = self.population_nodes_degrees_urban
        else:
            tmp = self.population_nodes_degrees_rural

        for i in range(1 if num_age_groups > 1 else 0, num_age_groups):
            if plot_s:
                label = f"Контакты: {i}, размер: {tmp[i]}" if num_age_groups > 1 else "Восприимчивые"
                plt.plot(self.time_linspace, self.solution[:, i * 3], c=cmap(norm(i if num_age_groups > 1 else i + 0)),
                         label=label)
            if plot_i:
                label = f"Контакты: {i}, размер: {tmp[i]}" if num_age_groups > 1 else "Инфицированные"
                plt.plot(self.time_linspace, self.solution[:, i * 3 + 1], c=cmap(norm(i if num_age_groups > 1 else i + 1)),
                         label=label)
            if plot_r:
                label = f"Контакты: {i}, размер: {tmp[i]}" if num_age_groups > 1 else "Выздоровевшие"
                plt.plot(self.time_linspace, self.solution[:, i * 3 + 2], c=cmap(norm(i if num_age_groups > 1 else i + 2)),
                         label=label)

        axs.set_xlabel(xlabel, fontdict={'fontsize': 18})
        axs.set_ylabel(ylabel, fontdict={'fontsize': 18})
        axs.set_title(title, fontdict={'fontsize': 18})

        if ylim != -1:
            axs.set_ylim(0, ylim)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # 2.2 рисуем colorbar только если не идет моделирование с одной группой
        if num_age_groups > 1:
            cbar = plt.colorbar(sm, label="Номер компартмента", ax = axs)
            cbar.set_ticks(0 + (np.arange(1, num_age_groups) - 0.5))
            cbar.set_ticklabels(list(range(1, num_age_groups)))
            #cbar.set_ticks(0 + (np.arange(num_age_groups) + 0.5))
            #cbar.set_ticklabels(list(range(num_age_groups)))
        plt.legend()
        plt.show()

        if save_path != "":
            fig.savefig(save_path, bbox_inches='tight')

        return 0

    def calc_main_statistics(self):
        """ Метод считает статистику по решенному дифференциальному уравнению """

        # 1. вычислить число компартментов
        num_age_groups = self.beta_normalized.size

        def _convert_linspace_to_days(linspace_period):
            return linspace_period * self.time_end // self.time_linspace.size

        infected_number, infection_moment_of_change, duration = {}, {}, {}
        duration_threshold = 0.5
        for i in range(1 if num_age_groups > 1 else 0, num_age_groups):
            # запомнить для каждого компартмента максимальное число зараженных
            infected_number[i] = self.solution[:, i * 3 + 1].max()
            # запомнить для каждого компартмента день, в который начался спад заражений
            infection_moment_of_change[i] = _convert_linspace_to_days(self.solution[:, i * 3 + 1].argmax())
            # запомнить для каждого компартмента, сколько дней длилась инфекция
            duration[i] = _convert_linspace_to_days(np.where(self.solution[:, i * 3 + 1] > duration_threshold)[0][-1])

        if self.sir_model_population_type == "urban":
            compartment_sizes = self.population_nodes_degrees_urban
        else:
            compartment_sizes = self.population_nodes_degrees_rural

        tmp = pd.DataFrame(np.array([np.array([*infected_number.keys()]), np.array([*infected_number.values()])]))
        tmp = pd.concat([tmp, pd.DataFrame([np.array([*duration.values()])])])
        tmp = pd.concat([tmp, pd.DataFrame([compartment_sizes[1:]])])
        tmp = pd.concat([tmp, pd.DataFrame([np.array([*infection_moment_of_change.values()])])])
        tmp.index = ["Число контактов  группы", "Максимальное число  инфицированных",
                     "Длительность  вспышки", "Размер  компартмента",
                     "Число дней, через которое начался спад заражений"]

        # 3. рассчитать агрегированные метрики
        # 3.1 Рассчитать средневзвешенную по размерам групп продолжительность вспышки
        tmp["Продолжительность (ср.взвш)"] = \
            np.sum(tmp.loc["Длительность  вспышки", :] * tmp.loc["Размер  компартмента", :]) \
            / np.sum(tmp.loc["Размер  компартмента", :])

        # 3.2 Рассчитать, сколько всего людей переболело
        tmp["Общее число переболевших"] = int(np.sum(tmp.loc["Максимальное число  инфицированных", :][:-1]))

        # 3.3 Рассчитать, сколько процентов людей переболело
        tmp["Процент переболевших"] = tmp["Общее число переболевших"] /compartment_sizes.sum() * 100

        # 3.4 Рассчитать средневзвешенную по размерам групп продолжительность острой фазы вспышки
        tmp["Продолжительность острой фазы (ср.взвш)"] = \
            np.sum(tmp.loc["Число дней, через которое начался спад заражений", :][:-3] * tmp.loc["Размер  компартмента", :][:-3]) \
            / np.sum(tmp.loc["Размер  компартмента", :][:-3])

        return tmp

    def plot_sir_model_curve(self,
                             population_size: int = 1000,
                             beta: np.ndarray = np.array([0.2]),  # коэффициент контактов [0, 1]
                             gamma: float = 1. / 10,              # коэффициент выздоровления в 1/ (число дней)
                             plot_s: bool = True,                 # рисовать ли кривую для восприимчивых
                             plot_i: bool = True,                 # рисовать ли кривую для инфицированных
                             plot_r: bool = True,                 # рисовать ли кривую для выздоровевших
                             n_days: int = 160,                   # промежуток дней, за которых рисовать график
                             i0: int = 1,                         # начальные значения для зараженных и выздоровевших
                             r0: int = 0,
                             beta_0: float = 1.0,                 # значение beta_0 для системы. Для каждой группы это значение домножается на соотв. beta[i]
                             use_generated_population_nodes_degrees=False,
                             population_type="urban",
                             plot_single_group: int = -1,
                             title="",
                             save_path: str = "",
                             use_label: bool = True,
                             ylim: int = -1) -> int:

        # Система дифференциальных уравнения SIR
        def deriv(y, t, N, beta, beta_0,  gamma, i):
            S, I, R = y
            dSdt = -beta_0 * beta[i] * S * I / N
            dIdt = beta_0 * beta[i] * S * I / N - gamma * I
            dRdt = gamma * I
            return dSdt, dIdt, dRdt

        if use_generated_population_nodes_degrees:

            if population_type == "urban":
                population_nodes_degrees = self.population_nodes_degrees_urban
            if population_type == "rural":
                population_nodes_degrees = self.population_nodes_degrees_rural

            if population_nodes_degrees is None:

                if population_type == "urban":
                    tmp = self.connections_matrix_urban.toarray()
                elif population_type == "rural":
                    tmp = self.connections_matrix_rural.toarray()
                else:
                    raise Exception

                # исключить контакты людей самих с собой
                np.fill_diagonal(tmp, 0)

                # создать граф для выбранных людей и рассчитать степени вершин
                graph = nx.Graph(tmp)
                population_nodes_degrees = nx.degree_histogram(graph)

            population_nodes_degrees = np.array(population_nodes_degrees)
            self.population_nodes_degrees_absolute_values = population_nodes_degrees.copy()

            tmp = np.arange(population_nodes_degrees.size)
            self.beta_inner = (tmp - tmp.min()) / (tmp.max() - tmp.min())
        else:
            self.beta_inner = beta

        self.betta_dict = dict(np.array([np.arange(population_nodes_degrees.size), self.beta_inner]).T)

        # получить средневзвешенную бету по всем группам
        self.mean_weight_beta = (np.array(list(self.betta_dict.keys())) * np.array(list(self.betta_dict.values()))) \
                                    .sum() / np.array(list(self.betta_dict.keys())).sum()

        # посчитать, сколько людей вообще переболело
        total_ever_was_infected = 0
        fig = plt.figure()
        ax = fig.add_subplot()

        t = np.linspace(0, n_days, n_days)

        cmap = plt.get_cmap('jet', np.unique(self.beta_inner).size)

        # хотим так же отследить момент на какой день приходится пик заражения
        self.moment_of_change_dict = dict()
        # хотим отследить длительность вспышки
        max_infection_duration = 0
        ticks = []

        # Normalizer
        norm = colors.Normalize(vmin=0,
                                vmax=self.beta_inner.size)

        for i, b in enumerate(self.beta_inner):


            # если нужно нарисовать только один конкретный график для демонстрации
            if plot_single_group != -1 and plot_single_group != i:
                continue

            # сколько в популяции людей с числом контактов i
            if use_generated_population_nodes_degrees:
                group_size = self.population_nodes_degrees_absolute_values[i]
            else:
                group_size = population_size

            s0 = group_size - i0 - r0
            y0 = s0, i0, r0

            if group_size == 0:
                continue

            ticks.append(i)
            # решить систему ОДУ для модели SIR
            S, I, R = odeint(deriv, y0, t, args=(group_size, self.beta_inner, beta_0, gamma, i)).T

            self.moment_of_change_dict[i] = np.argmax(I)
            if np.where(I > 0.9)[0][-1] > max_infection_duration:
                max_infection_duration = np.where(I > 0.9)[0][-1]

            if plot_s:
                ax.plot(t, S, alpha=0.5,  c=cmap(norm(i)),
                        label='Susceptible' if (i == 0 or plot_single_group != -1) and use_label else None)
            if plot_i:
                ax.plot(t, I, alpha=0.5, c=cmap(norm(i)),
                        label='Infected' if (i == 0 or plot_single_group != -1) and use_label else None)
            if plot_r:
                ax.plot(t, R,   alpha=0.5, c=cmap(norm(i)),
                        label='Recovered' if (i == 0 or plot_single_group != -1) and use_label else None)

            # запомнить, сколько людей в этой группе болело
            total_ever_was_infected += I.max()


        # creating ScalarMappable
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        plt.colorbar(sm)

        if ylim != -1:
            ax.set_ylim(0, ylim)
        ax.set_xlabel('Дни')
        ax.set_ylabel('Количество')
        ax.set_title(title, fontdict={'fontsize': 12})
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid()
        if use_label:
            legend = ax.legend()
            legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)

        if save_path != "":
            fig.savefig(save_path, bbox_inches='tight')

        infected = int(total_ever_was_infected)
        total = self.population_nodes_degrees_absolute_values.sum()

        # вычислить среднезвешенное по размерам групп значение пика инфеции (кол-во дней)
        group_size = population_nodes_degrees[list(self.moment_of_change_dict.keys())]
        moment_of_change = list(self.moment_of_change_dict.values())
        mean_weight_moment_of_chage = (np.array(group_size) * np.array(moment_of_change)).sum() // group_size.sum()
        plt.show()
        if plot_single_group == -1:
            print("--- Основная статистика ---")
            print(f"Всего было заражено {infected} людей из {total} ({np.round(infected / total * 100, 2)}%)")
            print("пик заражения по группам", mean_weight_moment_of_chage)
            print("Длительность ", max_infection_duration)

        return 0

    def plot_households_distribution(self,
                                     save_path: str = "") -> int:

        households_grouped = self.households_distribution_template[["region_type", "1_person", "2_persons",
                                                                    "3_persons", "4_persons", "5_persons",
                                                                    "6+_persons"]] \
                                 .groupby("region_type") \
                                 .sum() \
                                 .reset_index()

        plt.figure(figsize=(20, 7))
        fig, axs = plt.subplots(1, 2, figsize=(20, 7))
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

        sns.barplot(x='variable', y='value', data=df_melted_rural, ax=axs[1], color='red')
        sns.barplot(x='variable', y='value', data=df_melted_urban, ax=axs[0], color='red')

        axs[0].tick_params(axis='both', labelsize=11)
        axs[0].set_title("Городское население", fontdict={'fontsize': 20})
        axs[0].set_xlabel("Число людей в семье", fontdict={'fontsize': 14})
        axs[0].set_ylabel("Количество семей", fontdict={'fontsize': 14})

        axs[1].tick_params(axis='both', labelsize=11)
        axs[1].set_title("Сельское население", fontdict={'fontsize': 20})
        axs[1].set_xlabel("Число людей в семье", fontdict={'fontsize': 14})
        axs[1].set_ylabel("Количество семей", fontdict={'fontsize': 14})

        if save_path != "":
            plt.savefig(save_path, bbox_inches='tight')

        return 0

    def generate_total_population(self,
                                  population_size: int,
                                  largest_manufactures_number: int = 20,
                                  lockdown: bool = False,
                                  use_small_world_approach: bool = False,
                                  use_random_connections_approach: bool = False,
                                  random_connections_constant: float = 0.3,
                                  betta: float = 0.2,
                                  input_folder: str = "input",
                                  output_folder: str = "output") -> int:

        """ Метод создает городскую и сельскую популяции """

        # 1. создание городской популяции
        print(datetime.datetime.now(), ": Строится популяция для городского населения ... ")
        self.create_population(population_type="urban",
                               population_size=(population_size // 2),
                               largest_manufactures_number=largest_manufactures_number,
                               lockdown=lockdown,
                               use_small_world_approach=use_small_world_approach,
                               use_random_connections_approach=use_random_connections_approach,
                               random_connections_constant=random_connections_constant,
                               betta=betta,
                               input_folder=input_folder,
                               output_folder=output_folder)

        urban_population_raw = self.population.copy()
        self.connections_matrix_urban = self.connections_matrix.copy()
        urban_population = self.population.groupby(["age_group", "sex", "population_type"],
                                                   as_index=False).count()

        # 2. создание сельской популяции
        print(datetime.datetime.now(), ": Строится популяция для сельского населения ... ")
        self.create_population(population_type="rural",
                               population_size=(population_size // 2),
                               largest_manufactures_number=largest_manufactures_number,
                               lockdown=lockdown,
                               use_small_world_approach=use_small_world_approach,
                               use_random_connections_approach=use_random_connections_approach,
                               random_connections_constant=random_connections_constant,
                               betta=betta,
                               input_folder=input_folder,
                               output_folder=output_folder)

        rural_population_raw = self.population.copy()
        self.connections_matrix_rural = self.connections_matrix.copy()
        rural_population = self.population.groupby(["age_group", "sex", "population_type"],
                                                   as_index=False).count()

        # 3. соединение городской и сельской популяций в одну
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

        # 4. формирование степеней вершин для городского и сельского населений
        tmp = self.connections_matrix_urban.toarray()
        np.fill_diagonal(tmp, 0)
        self.population_nodes_degrees_urban = np.array(nx.degree_histogram(nx.Graph(tmp)))

        tmp = self.connections_matrix_rural.toarray()
        np.fill_diagonal(tmp, 0)
        self.population_nodes_degrees_rural = np.array(nx.degree_histogram(nx.Graph(tmp)))

        return 0

    def plot_households_distribution_generated_population(self,
                                                          ylim: int = 5000,
                                                          save_path: str = "") -> int:
        """ Метод рисует распределение по домохозяйствам для сгенерированной популяции"""
        if self.population_total is None:
            print("Нет данных для отрисовки")
            raise Exception

        tmp = self.population_total \
                  .query("population_type == 'urban'") \
                  .groupby("household_id") \
                  .count() \
                  .groupby("id") \
                  .count() \
                  .reset_index()[["id", "age"]].head(6).transpose()

        tmp.columns = tmp.loc["id", :]
        tmp = tmp.tail(1)
        tmp_urban = tmp.melt()

        tmp = self.population_total.query("population_type == 'rural'").groupby("household_id").count().groupby(
            "id").count().reset_index()[["id", "age"]].head(6).transpose()
        tmp.columns = tmp.loc["id", :]
        tmp = tmp.tail(1)
        tmp_rural = tmp.melt()

        plt.figure(figsize=(20, 7))
        fig, axs = plt.subplots(1, 2, figsize=(20, 7))
        plt.style.use('ggplot')

        axs[0].set_ylim(0, ylim)
        axs[1].set_ylim(0, ylim)

        sns.barplot(x='id', y='value', data=tmp_rural, ax=axs[1], color='red')
        sns.barplot(x='id', y='value', data=tmp_urban, ax=axs[0], color='red')

        axs[0].set_title("Городское население", fontdict={'fontsize': 22})
        axs[0].set_xlabel("Число людей в семье", fontdict={'fontsize': 18})
        axs[0].set_ylabel("Количество семей", fontdict={'fontsize': 18})
        axs[0].set_xticks(ticks=tmp_urban.index, labels=["1 человек", "2 человека",
                                                         "3 человека", "4 человека",
                                                         "5 человек", "Более 6 человек"])
        axs[0].tick_params(axis='both', labelsize=11)

        axs[1].set_title("Сельское население", fontdict={'fontsize': 22})
        axs[1].set_xlabel("Число людей в семье", fontdict={'fontsize': 18})
        axs[1].set_ylabel("Количество семей", fontdict={'fontsize': 18})
        axs[1].set_xticks(ticks=tmp_rural.index, labels=["1 человек", "2 человека",
                                                         "3 человека", "4 человека",
                                                         "5 человек", "Более 6 человек"])
        axs[1].tick_params(axis='both', labelsize=11)

        if save_path != "":
            fig.savefig(save_path, bbox_inches='tight')

        return 0

    def plot_age_sex_distribution(self,
                                  max_xlimit: int,
                                  population_size: int,
                                  use_generated_population: bool = False,
                                  age_sex_distribution_filename: str = "age_sex_distribution_percentage.xlsx",
                                  save_path: str = "") -> int:
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
                print("Задан режим построения пирамиды по существующей популяции, но она не была сгенерирована")
                raise Exception

            age_sex_distribution_template_inner = self.population_total_grouped

        # 2. Отрисовка графика
        plt.figure(figsize=(20, 10))
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        axs[0].set_xlim(-max_xlimit, max_xlimit)
        axs[1].set_xlim(-max_xlimit, max_xlimit)

        AgeClass = age_sex_distribution_template_inner["age_group"][::-1]

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

        axs[0].set_title("Городское население", fontdict={'fontsize': 20})
        axs[0].set_xlabel("Численность")
        axs[0].set_ylabel("Возрастная группа")

        axs[1].set_title("Сельское население", fontdict={'fontsize': 20})
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

        if save_path != "":
            fig.savefig(save_path, bbox_inches='tight')

        plt.show()
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
                                           weight: int = 10,
                                           betta: float = 0.2) -> int:
        """ Функция создает связи внутри классов по школам """

        average_class_size = int(average_school_size // 11)

        number_of_schools = int(self.population.query("age <= 18").shape[0] * 0.75 / average_school_size)

        self.population["school_number"] = -1

        # создать словарь с индексами по возрастным категориям
        for j in range(number_of_schools):
            for age in range(7, 19, 1):
                # выбрать детей, которые будут ходить в школу
                # из числа детей, подходящих по возрасту
                ind = np.array(self.population.query("(age == @age) & (school_number == -1)")["id"].index)
                students = np.random.choice(ind, min(average_class_size, ind.size), replace=False)

                if len(students) <= 1:
                    continue

                # закрепить за этими детьми школу
                self.population.loc[students, "school_number"] = j

                # сформировать между выбранными людьми связи по модели small-world
                self._create_small_world_connections(ind=students, betta=betta)

        return 0

    def plot_manufactures_size_distribution(self, save_path: str = "") -> Literal[0, 1]:
        """ Функция рисует, как распределены предприятия по размерам """

        if self.display_status:
            print(datetime.datetime.now(), ": Рисуется распределение домохозяйств по размеру ... ")

        fig, ax = plt.subplots(figsize=(10, 10))
        keys, values = list(self.manufactures_sizes_dict.keys()), list(self.manufactures_sizes_dict.values())

        # поставить логарифмическую шкалу
        ax.set_xscale('log')

        # построить график
        ax.tick_params(axis='both', labelsize=14)
        ax.plot(keys, values)
        ax.scatter(keys, values)
        ax.grid(True)
        ax.set_xlabel("Размер предприятия (число сотрудников)", fontdict={'fontsize': 16})
        ax.set_ylabel("Количество предприятий", fontdict={'fontsize': 16})
        ax.set_title("Распределение предприятий по размеру", fontdict={'fontsize': 20})
        plt.show()

        if save_path != "":
            fig.savefig(save_path, bbox_inches='tight')

        return 0

    def plot_manufactures_connections_graph(self, departament_id: int = -1,
                                            save_path: str = "") -> Literal[0, 1]:
        """ Функция рисует граф контактов внутри предприятия"""

        np.random.seed(self.random_seed)

        if departament_id == -1:
            departament_id_inner = np.random.choice(np.array(self.population["departament_number"] \
                                                                 [self.population["departament_number"] != -1]), 1)[0]
        else:
            departament_id_inner = departament_id

        # получить id людей, которые работают на этом предприятии
        ind = self.population.query("departament_number == @departament_id")["id"]

        # создать граф
        tmp = self.connections_matrix[ind, :].tocsc()[:, ind].toarray()

        fig = plt.figure()
        np.fill_diagonal(tmp, 0)
        graph = nx.DiGraph(tmp)
        pos = nx.circular_layout(graph)

        nx.draw(G=graph, pos=pos)
        plt.show()

        if save_path != "":
            fig.savefig(save_path, bbox_inches='tight')

        return 0

    def plot_total_connections_hist(self,
                                    population_type: Literal["urban", "rural"] = None,
                                    save_path: str = "",
                                    title: str = "") -> Literal[0, 1]:
        """ Нарисовать гистограмму распределения степеней вершин для всего графа"""

        if population_type is None:
            print("Ошибка: Нет матрицы контактов")
            return 1

        if population_type not in ["urban", "rural"]:
            print("Ошибка: Недопустимый тип популяции")
            return 1

        # 1. нарисовать гистограмму распределения степеней вершин для заданной популяции
        fig, ax = plt.subplots(figsize=(10, 10))
        tmp = pd.DataFrame(self.population_nodes_degrees_urban if population_type == "urban"
                           else self.population_nodes_degrees_rural, columns=["tmp"]).reset_index()
        sns.barplot(x='index', y='tmp', data=tmp, ax=ax, color='red')

        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel('Степень вершины', fontdict={'fontsize': 18})
        ax.set_ylabel('Число вершин', fontdict={'fontsize': 18})
        ax.set_title(title, fontdict={'fontsize': 22})

        plt.show()

        if save_path != "":
            fig.savefig(save_path, bbox_inches='tight')

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
        graph = nx.Graph(tmp)

        hist = nx.degree_histogram(graph)

        # Рисуем гистограмму
        plt.bar(np.arange(len(hist), dtype=np.int8), hist, width=0.8, color='b')
        plt.xticks(np.arange(len(hist), dtype=np.int8), np.arange(len(hist), dtype=np.int8))
        plt.xlabel('Степень вершины')
        plt.ylabel('Число вершин')
        plt.title('Гистограмма степеней вершин для предприятия ' + str(manufacture_id_inner))
        plt.show()

        return 0

    def _create_connections_inside_manufactures(self,
                                                largest_manufactures_number: int,
                                                weight: int = 20,
                                                betta: float = 0.05,
                                                lockdown: bool = False) -> int:

        # определить число людей работоспособного возраста
        number_of_workers = self.population.query("age > 18").shape[0]

        # если введен локдаун, работает только 30 процентов населения
        if lockdown:
            number_of_workers *= 0.3

        self.population["manufacture_number"] = -1
        self.population["departament_number"] = -1

        # определить порядок числа number_of_workers
        order = int(np.floor(np.log10(number_of_workers)))

        a = np.power(10, np.arange(order))[::-1]

        m, iter_number = -1, 0
        while m <= 0:
            m = int((number_of_workers - a @ np.ones(a.size) * largest_manufactures_number) // (a @ np.arange(a.size)))
            if m <= 0:
                largest_manufactures_number -= 1
                if largest_manufactures_number <= 0:
                    raise Exception

                print(f"Значение параметра largest_manufactures_number было изменено на: {largest_manufactures_number}")
                if iter_number > 10:
                    raise Exception

                iter_number += 1

        self.manufactures_sizes_series = pd.Series(
            np.repeat(a, m * np.arange(a.size) + largest_manufactures_number)).value_counts()
        self.manufactures_sizes_dict = self.manufactures_sizes_series.to_dict()

        for manufacture_size in tqdm(self.manufactures_sizes_dict.keys()):
            for j in range(self.manufactures_sizes_dict[manufacture_size]):

                # выбрать людей трудоспособного возраста, за которыми не закреплено предприятие
                ind = self.population.query("(age >= 19) & (manufacture_number == -1)")["id"]

                # выбрать людей, которых закрепим за текущим предприятием
                workers_total = np.random.choice(ind, manufacture_size, replace=False)

                if not len(workers_total):
                    continue

                # закрепить за этими людьми номер предприятие
                self.population.loc[workers_total, "manufacture_number"] = j * manufacture_size

                # -- создать связи между людьми по схеме small-world --
                # Так как предприятия могут быть очень большими и в них может работать по несколько тысяч человек
                # было решено разбить каждое предприятие на департаменты и в них произвести моделирование по
                # алгоритму small-world

                departament_size = 20
                for k in range(len(workers_total) // departament_size):
                    if k != len(workers_total) // departament_size - 1:
                        workers = workers_total[k * departament_size: (k+1) * departament_size]
                    else:
                        workers = workers_total[k * departament_size:]

                    # закрепить за этими людьми номер предприятие
                    self.population.loc[workers, "departament_number"] = j * manufacture_size * k

                    self._create_small_world_connections(ind=workers, betta=betta)

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
                                   display_status: bool = True,
                                   population_type: Literal["urban", "rural"] = None) -> int:
        """ Функция рисует остовное дерево по матрице контактов """

        if display_status:
            print(datetime.datetime.now(), ": Создается граф по матрице контактов ... ")

        if population_type is None:
            print("Ошибка: Нет матрицы контактов")
            return 1

        if population_type not in ["urban", "rural"]:
            print("Ошибка: Недопустимый тип популяции")
            return 1

        matrix_inner = None
        if population_type == "urban":
            matrix_inner = self.connections_matrix_urban.toarray()

        if population_type == "rural":
            matrix_inner = self.connections_matrix_rural.toarray()

        np.fill_diagonal(matrix_inner, 0)

        mst = nx.minimum_spanning_tree(nx.from_numpy_array(matrix_inner))

        if display_status:
            print(datetime.datetime.now(), ": Выполняется отрисовка построенного графа ... ")

        nx.draw(mst, node_color='blue', node_size=10, edge_color='grey')
        plt.title('Остовное дерево')

        plt.show()
        return 0

    def plot_heat_map(self,
                      population_type: Literal["urban", "rural"] = None) -> int:
        """ Функция рисует тепловую карту матрицы контактов """
        if population_type is None:
            print("Ошибка: Нет матрицы контактов")
            return 1

        if population_type == "urban":
            plt.imshow(self.connections_matrix_urban.toarray(), cmap="hot", interpolation="nearest")

        if population_type == "rural":
            plt.imshow(self.connections_matrix_rural.toarray(), cmap="hot", interpolation="nearest")

        plt.colorbar()
        plt.show()
        return 0

    """
    def plot_contact_matrix(self,
                            population_type: Literal["urban", "rural"] = None) -> int:
 
        if population_type is None:
            print("Ошибка: Нет матрицы контактов")
            return 1

        inner_matrix =

        if population_type == "urban":
            inner_matrix = self.connections_matrix_urban.toarray()

        if population_type == "rural":
            inner_matrix = self.connections_matrix_rural.toarray()





        plt.colorbar()
        plt.show()
        return 0

    """
    def _create_connection_inside_university(self,
                                             average_university_size: int,
                                             average_number_of_groups: int,
                                             weight: int = 30,
                                             betta: float = 0.2) -> int:

        average_group_size = int(average_university_size // average_number_of_groups)
        number_of_universities = int(
            self.population.query("(age > 18) & (age < 27)").shape[0] / average_university_size)

        self.population["university_number"] = -1

        # создать словарь с индексами по возрастным категориям
        for k in tqdm(range(number_of_universities)):
            for department in range(average_number_of_groups):
                # выделить тех, кто подходит под характеристики студента
                ind = self.population.query("(age > 18) & (age < 27) & (university_number == -1)")["id"]
                students = np.random.choice(ind, min(average_group_size, ind.size), replace=False)

                if not len(students):
                    continue

                # закрепить за выбранными людьми университет
                self.population.loc[students, "university_number"] = k

                # сформировать между выбранными людьми связи по модели small-world
                self._create_small_world_connections(ind=students, betta=betta)

        return 0

    def _create_small_world_connections(self,
                                        ind: np.ndarray,
                                        betta: float) -> Literal[0, 1]:
        """ Метод создает контакты по модели small-world"""

        if self.connections_matrix is None:
            self.connections_matrix = lil_matrix((self.population.shape[0], self.population.shape[0]),
                                                 dtype=np.int8)

        # создать связи между людьми по схеме small-world
        number_of_nodes = ind.size

        # сделать так, чтобы изначально все люди стоят по кругу и соединены только с соседом
        for j in range(number_of_nodes - 1):
            self.connections_matrix[ind[j], ind[j + 1]] = 1
            self.connections_matrix[ind[j + 1], ind[j]] = 1
        self.connections_matrix[ind[0], ind[-1]] = 1
        self.connections_matrix[ind[-1], ind[0]] = 1

        # определить сколько будет перебрасываний связей
        number_of_ripped_edges = int(number_of_nodes * betta)

        # выделить студентов, которым будем менять связи
        index = np.random.choice(ind, number_of_ripped_edges, replace=False)

        for i in range(len(index) - 1):
            if i != index.size - 1:
                # разорвать связь
                self.connections_matrix[index[i], index[i + 1]] = 0
                self.connections_matrix[index[i + 1], index[i]] = 0
                # добавить новую случайную связь
                new_node = np.random.choice(index[index != index[i]], 1, replace=False)[0]
                self.connections_matrix[index[i], new_node] = 1
                self.connections_matrix[new_node, index[i]] = 1
            else:
                # разорвать связь
                self.connections_matrix[index[0], index[-1]] = 0
                self.connections_matrix[index[-1], index[0]] = 0
                # добавить новую случайную связь
                new_node = np.random.choice(index[index != index[-1]], 1, replace=False)[0]
                self.connections_matrix[index[-1], new_node] = 1
                self.connections_matrix[new_node, index[-1]] = 1

        return 0

    def create_population(self,
                          population_type: Literal["urban", "rural"],
                          population_size: int,
                          largest_manufactures_number: int = 20,
                          households_filename: str = "households.xlsx",
                          age_sex_distribution_filename: str = "age_sex_distribution_percentage.xlsx",
                          manufactures_filename: str = "manufactures.xlsx",
                          schools_filename: str = "schools.xlsx",
                          lockdown: bool = False,
                          use_small_world_approach: bool = False,
                          use_random_connections_approach: bool = False,
                          random_connections_constant: float = 0.3,
                          betta: float = 0.2,
                          input_folder: str = "input",
                          output_folder: str = "output") -> Literal[0, 1]:

        print(datetime.datetime.now(), ": Запуск функции создания популяции ... ")

        # считать все необходимые шаблоны для генерации
        self.read_households_distribution_template(file_name=input_folder + '/' + households_filename)
        self.read_age_sex_distribution_template(file_name=input_folder + '/' + age_sex_distribution_filename)
        self.read_manufactures_distribution_template(file_name=input_folder + '/' + manufactures_filename)
        self.read_schools_distribution_template(file_name=input_folder + '/' + schools_filename)

        # посчитать, сколько человек надо на каждый тип домохозяйства
        self._get_household_ratio()

        # найти число людей для каждого типа домохозяйства
        if population_type == "urban":
            _population_type = self.household_ratio_urban
        elif population_type == "rural":
            _population_type = self.household_ratio_rural
        else:
            print("Задано недопустимое значение для типа популяции")
            return 1

        households_number = _population_type * population_size // np.arange(1, 7, 1)

        # 1.1 найти число людей для каждой возрастной группы
        base = self.age_sex_distribution_template[["age", "men" + "_" + population_type,
                                                   "women" + "_" + population_type]] \
            .rename(
            columns={"men" + "_" + population_type: "men",
                     "women" + "_" + population_type: "women"}
        )

        # 1.2 считаем, что мужчин и женщин равное количество в популяции
        # определяем, сколько человек в каждой возрастной категории
        base.loc[:, ["men", "women"]] = base.loc[:, ["men", "women"]] * population_size * 0.5
        base.loc[base["age"] == '70 лет и более', "age"] = "70"
        base = base.astype({'men': int, 'women': int, "age": int})

        print(datetime.datetime.now(), ": Создается популяция мужчин на основе шаблона ... ")
        men = self._create_population_from_data(population_type="men", distribution_template=base)

        print(datetime.datetime.now(), ": Создается популяция женщин на основе шаблона ... ")
        women = self._create_population_from_data(population_type="women", distribution_template=base)

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
                                            ['0 – 4', '5 – 9', '10 – 14', '15 – 19', '20 – 24',
                                                     '25 – 29', '30 – 34', '35 – 39', '40 – 44', '45 – 49',
                                                     '50 – 54', '55 – 59', '60 – 64', '65 – 69', '70 лет и более'])

        self.population = population
        self.population["population_type"] = population_type

        # если задано условие моделирования в режиме small-world. При этом подходе внутри популяции
        # просто создаются контакты без разбивки на предприятия, домохозяйства, школы и университеты
        if use_small_world_approach:
            print(datetime.datetime.now(), ": Строится случайный граф по модели small-world ... ")
            self.population = self.population.reset_index(drop=True).reset_index().rename(columns={"index": "id"})
            self._create_small_world_connections(ind=np.array(self.population["id"].index), betta=betta)
            return 0

        if use_random_connections_approach:
            print(datetime.datetime.now(), ": Строится случайный граф Эрдёша-Реньи ... ")
            self.population = self.population.reset_index(drop=True).reset_index().rename(columns={"index": "id"})
            self.connections_matrix = nx.to_scipy_sparse_array(nx.erdos_renyi_graph(population_size,
                                                                                    random_connections_constant),
                                                               format='lil')
            return 0

        # 2.1 разбить полученную популяцию на домохозяйства
        print(datetime.datetime.now(), ": Формирование домохозяйств ... ")
        self._create_connections_in_population(households_number=pd.Series(data=households_number.astype("int"),
                                                                           index=['1_person', '2_persons',
                                                                                  '3_persons', '4_persons',
                                                                                  '5_persons', '6+_persons']))

        # 3 создать матрицу контактов для внутри домохозяйств
        print(datetime.datetime.now(), ": Создание матрицы контактов внутри домохозяйств ... ")
        self._create_contacts_inside_households(weight=10)

        # 4 создать матрицу контактов внутри школ
        if lockdown and population_type == "urban":
            print(datetime.datetime.now(), ": Связи внутри школ не создаются, как так введен локдаун ... ")
        else:
            print(datetime.datetime.now(), ": Создается контакты внутри школ ... ")
            average_school_size = int(self.schools_distribution_template["Число обучающихся на одну школу"].mean())
            self._create_connections_inside_schools(average_school_size=average_school_size,
                                                    weight=10,
                                                    betta=betta)

        # 5 создать матрицу контактов внутри предприятий
        if lockdown and population_type == "urban":
            print(datetime.datetime.now(), ": Контакты внутри предприятий создаются только для 30% популяции ... ")
            self._create_connections_inside_manufactures(largest_manufactures_number=largest_manufactures_number,
                                                         weight=10,
                                                         lockdown=lockdown,
                                                         betta=betta)
        else:
            print(datetime.datetime.now(), ": Создается контакты внутри предприятий ... ")
            self._create_connections_inside_manufactures(largest_manufactures_number=largest_manufactures_number,
                                                         weight=10,
                                                         lockdown=lockdown,
                                                         betta=betta)

        # 6 создать связи внутри университетов
        if lockdown and population_type == "urban":
            print(datetime.datetime.now(), ": Связи внутри университетов не создаются, как так введен локдаун ... ")
        else:
            print(datetime.datetime.now(), ": Создается контакты внутри университетов ... ")
            self._create_connection_inside_university(average_university_size=300,
                                                      average_number_of_groups=30,
                                                      weight=10,
                                                      betta=betta)

        # найти число людей для каждого типа домохозяйства
        if population_type == "urban":
            self.connections_matrix_urban = self.connections_matrix.copy()
        elif population_type == "rural":
            self.connections_matrix_rural = self.connections_matrix.copy()
        else:
            raise Exception

        print(datetime.datetime.now(), ": Создание популяции завершено ... ")

        return 0
