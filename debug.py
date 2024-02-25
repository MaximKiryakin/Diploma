
import load_data as ld

import graph as gr
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
p = gr.MyPopulation()
p.create_population("urban",50000 )
#p.plot_graph()
#p.plot_manufactures_size_distribution()
p.plot_manufactures_connections_hist()

