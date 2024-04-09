
import load_data as ld

import graph as gr
import pandas as pd
import load_data as ld
import seaborn as sns
import graph as gr

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np
import warnings

import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
"""
households = ld.read_households_distribution_template(file_name="households.xlsx")
age = ld.read_age_sex_distribution_template(file_name="age_sex_distribution_percentage.xlsx")
manufactures = ld.read_manufactures_distribution_template(file_name="manufactures.xlsx")
schools = ld.read_schools_distribution_template(file_name="schools.xlsx")

gr.create_population(households_distribution_template=households,
                     age_sex_distribution_template=age,
                     population_type="urban",
                     population_size=10000,
                     schools_distribution_template=schools)
"""
"""
p = gr.MyPopulation()
p.create_population("urban",20000, largest_manufactures_number=12)
#p.plot_graph()
#p.plot_heat_map()
#p.plot_manufactures_size_distribution()
#p.plot_manufactures_connections_hist()
p.plot_total_connections_hist()
"""

population_size = 20000
model = gr.MyPopulation()
model.generate_total_population(population_size = population_size,
                                largest_manufactures_number=13,
                                lockdown=False)
