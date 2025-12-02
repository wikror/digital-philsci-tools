import pandas as pd
import json


DATA_PATH = f"/home/wikror/gdrive/corpus-study/data/"
query_name = "results-communication-def-separate"

DISCIPLINES = ['animalBehavior', 'neuroscience', 'psychology', 'developmentalBiology', 'ecologyEvolution', 'microbiology', 'molecularBiology', 'plantScience', 'varia']

incl_df = pd.read_json(f"{DATA_PATH}{query_name}-incl.json")
# excl_df = pd.read_json(f"{DATA_PATH}{query_name}-excl.json")

incl_df_exploded = incl_df.explode(["query", "distance"])
# excl_df_exploded = excl_df.explode(["query", "distance"])

stats_df = pd.DataFrame.from_dict({idx: {"mean": group.distance.mean(), 
                                        "std": group.distance.std(), 
                                        "min": group.distance.min(), 
                                        "median":group.distance.median(), 
                                        "max": group.distance.max()
                                        } for idx, group in incl_df_exploded.groupby("query")}, orient="index")
"""
{
    "0": {
        "mean": 0.3733192497116256,
        "std": 0.10993303506293543,
        "min": 0.2135394216,
        "median": 0.33119082450000004,
        "max": 0.6995950937000001
    },
    "1": {
        "mean": 0.4156738154047867,
        "std": 0.07889328636469592,
        "min": 0.283903569,
        "median": 0.39672374725000004,
        "max": 0.6676392555
    },
    "2": {
        "mean": 0.41752827150967403,
        "std": 0.11703006458084306,
        "min": 0.2734505534,
        "median": 0.3680096269,
        "max": 0.7659111023
    },
    "3": {
        "mean": 0.4430962622285596,
        "std": 0.06153226960754399,
        "min": 0.3387622535,
        "median": 0.4368798137,
        "max": 0.7919757962
    },
    "4": {
        "mean": 0.4653323865315861,
        "std": 0.08082584828691529,
        "min": 0.3269543052,
        "median": 0.47471401100000005,
        "max": 0.8422043324
    },
    "5": {
        "mean": 0.46789380002722103,
        "std": 0.07943715441124892,
        "min": 0.343341589,
        "median": 0.4462479949,
        "max": 0.8602485657000001
    },
    "6": {
        "mean": 0.4311827487768396,
        "std": 0.10598525154008541,
        "min": 0.2865785062,
        "median": 0.39264628290000003,
        "max": 0.7554420829
    },
    "7": {
        "mean": 0.4340707290416657,
        "std": 0.06195652574490445,
        "min": 0.31552553180000004,
        "median": 0.43706171215,
        "max": 0.6640536785000001
    },
    "8": {
        "mean": 0.471690831078888,
        "std": 0.12126479610184375,
        "min": 0.3204491735,
        "median": 0.42215147610000003,
        "max": 0.7872999907
    },
    "9": {
        "mean": 0.41270479268452565,
        "std": 0.06553424418410346,
        "min": 0.3188458085,
        "median": 0.3936103284,
        "max": 0.6834412813
    },
    "10": {
        "mean": 0.43362672308803074,
        "std": 0.05407437805166734,
        "min": 0.3112770915,
        "median": 0.4395027757,
        "max": 0.6288302541
    },
    "11": {
        "mean": 0.4325107643070924,
        "std": 0.12080962542380437,
        "min": 0.2895347178,
        "median": 0.37830445170000004,
        "max": 0.7911772728
    },
    "12": {
        "mean": 0.4590886486128852,
        "std": 0.08867599507370429,
        "min": 0.2885684371,
        "median": 0.44309818745,
        "max": 0.7242184281
    },
    "13": {
        "mean": 0.4037499748813602,
        "std": 0.11592959953627076,
        "min": 0.2626172304,
        "median": 0.3545643985,
        "max": 0.7475219369
    },
    "14": {
        "mean": 0.4249502411332725,
        "std": 0.036325083237305,
        "min": 0.3587149978,
        "median": 0.41893458365,
        "max": 0.5953086615000001
    },
    "15": {
        "mean": 0.3917184231449073,
        "std": 0.03090414943172058,
        "min": 0.3355791271,
        "median": 0.38590925930000003,
        "max": 0.5324092507
    },
    "16": {
        "mean": 0.4174549132992396,
        "std": 0.08826865674035222,
        "min": 0.2940690815,
        "median": 0.3922194839,
        "max": 0.84017241
    },
    "17": {
        "mean": 0.4137679983092711,
        "std": 0.10612503494329507,
        "min": 0.2756278813,
        "median": 0.37161961200000004,
        "max": 0.7100136876
    },
    "18": {
        "mean": 0.45655075625293046,
        "std": 0.10835800830563581,
        "min": 0.3121284246,
        "median": 0.41492804885,
        "max": 0.7668462396
    },
    "19": {
        "mean": 0.43778948669715384,
        "std": 0.06764220349649298,
        "min": 0.3092351854,
        "median": 0.43370389940000004,
        "max": 0.6818174720000001
    },
    "20": {
        "mean": 0.4784723390108891,
        "std": 0.07473798470319956,
        "min": 0.3113135099,
        "median": 0.482773602,
        "max": 0.7702558637
    },
    "21": {
        "mean": 0.4242360806066616,
        "std": 0.06884758628607796,
        "min": 0.27703890200000003,
        "median": 0.43140111864999997,
        "max": 0.6736981869
    },
    "22": {
        "mean": 0.4471839671090839,
        "std": 0.050484721624425744,
        "min": 0.3286750913,
        "median": 0.44742003080000003,
        "max": 0.6864410639
    },
    "23": {
        "mean": 0.43347694686855026,
        "std": 0.07920524578671033,
        "min": 0.267680794,
        "median": 0.4498203397,
        "max": 0.720084846
    },
    "24": {
        "mean": 0.4407845690823243,
        "std": 0.05744887020581505,
        "min": 0.3100717068,
        "median": 0.4479436874,
        "max": 0.7174004316
    },
    "25": {
        "mean": 0.47433563834683945,
        "std": 0.05384223145542582,
        "min": 0.3626701832,
        "median": 0.47823691370000004,
        "max": 0.7173461318000001
    },
    "26": {
        "mean": 0.4598504161859913,
        "std": 0.07614119662706968,
        "min": 0.2956184149,
        "median": 0.4716264606,
        "max": 0.7475668788000001
    },
    "27": {
        "mean": 0.4572321065500648,
        "std": 0.054118880625869586,
        "min": 0.3668632209,
        "median": 0.4488227367,
        "max": 0.6763162017000001
    },
    "28": {
        "mean": 0.4327829185391175,
        "std": 0.06031333640602557,
        "min": 0.2972990274,
        "median": 0.44064468145,
        "max": 0.6914156079
    }
}
"""


# construct a graph to visualize relationships between queries: nodes are query texts, 
# edges are drawn when the same result was returned by the two queries, with weight corresponding to the amount of shared results

import networkx as nx
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

G = nx.Graph()
G.add_nodes_from(list(range(0, 28)))
edges = [tup for qs in incl_df["query"] for tup in list(combinations(qs, 2))]
edges_weights = [(k[0], k[1], {"weight": v}) for k, v in Counter(edges).items()]
G.add_edges_from(edges_weights)
nx.draw(G, with_labels=True)
plt.savefig(f"{DATA_PATH}graph.png")

deg = pd.DataFrame.from_dict(G.degree(weight="weight"))
deg.sort_values(1)


# check for correlation between topics and queries
from scipy.stats import chi2_contingency
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json

plt.rcParams['text.usetex'] = True

df = pd.read_csv(f"/home/wikror/gdrive/corpus-study/data/document-info-results-communication-def-separate-incl-nodups-50.csv")
new = df[["query", "Topic"]]
new["query"] = new["query"].apply(json.loads)
new = new.explode("query")
query_cl, _ = new["query"].factorize()
topic_cl, _ = new["Topic"].factorize()
contingency_tab = pd.crosstab(index=query_cl, columns=topic_cl)
chi2_res = chi2_contingency(contingency_tab)
print(chi2_res)
ax = sns.heatmap(contingency_tab, cmap="Reds", xticklabels=True, yticklabels=True)
ax.set(xlabel="Topics", ylabel="Queries")
plt.savefig("/home/wikror/gdrive/corpus-study/data/com-def-50-topics-queries-heatmap.png", dpi=300)
plt.clf()
del ax
ax = sns.clustermap(contingency_tab, cmap="Reds", xticklabels=True, yticklabels=True)
ax.ax_heatmap.set(xlabel="Topics", ylabel="Queries")
plt.savefig("/home/wikror/gdrive/corpus-study/data/com-def-50-topics-queries-clustermap.png", dpi=300)