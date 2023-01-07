import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt

database_filepath = "C:/Users/Lisann/Documents/Studium/2022_2023 WS RWTH/Data Science Course/" \
                    "CapstoneProject/data/PredicitiveMaintentace"
model_filepath = "LinearCSV"

# load data from database
engine = create_engine('sqlite:///' + database_filepath)
df = pd.read_sql_table('PredicitiveMaintentace', engine)

# extract data with only the power failure or no failure
df_powerfa = df[(df["Failure Type"] == "Power Failure") | (df["Failure Type"] == "No Failure")]
df_powerfa = df_powerfa.assign(powerkW=lambda d: d["Power [W]"].div(1000))
df_powerfa = df_powerfa.rename({"powerkW": "Power [kW]"}, axis=1)
print(df_powerfa)
# separate data into above and below median to get both failures from low and high values
df_above = df_powerfa[df_powerfa["Power [kW]"] >= df_powerfa["Power [kW]"].median()]
df_below = df_powerfa[df_powerfa["Power [kW]"] <= df_powerfa["Power [kW]"].median()]

plt.rcParams.update({'font.size': 13})
# show the failure area visualized in a boxplot
plt.figure(figsize = (13, 8))
plt.subplot(1, 2, 1)
sns.boxplot(data=df_above, y="Power [kW]", x="Target")
plt.subplot(1, 2, 2)
sns.boxplot(data=df_below, y="Power [kW]", x="Target")
plt.savefig("boxplot_powerfailure.jpg", bbox_inches='tight')
#plt.show()

plt.figure(figsize = (10, 8))
sns.countplot(data=df, x="Target")
plt.savefig("amount_target.jpg", bbox_inches='tight')
#plt.show()

df_notarget = df.drop(columns=["Target"])
plt.figure(figsize = (20, 15))
#sns.pairplot(df, hue="Target")
sns.pairplot(df_notarget, hue="Failure Type")
plt.savefig("pairplot_failuretype.jpg")
#plt.show()

plt.figure(figsize = (13, 13))
sns.heatmap(df.corr(), annot = True, fmt = ".2f")
plt.savefig("heatmap_features.jpg", bbox_inches='tight')