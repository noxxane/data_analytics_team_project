"""data analytics team project analyzing ai data"""

from datetime import datetime
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

START_DATE = "2013-1-1"

training_compute = pd.read_csv(
    "/home/nox/coding/data_analytics_team_project/artificial-intelligence-training-computation.csv"
)
training_compute = training_compute.sort_values(by="Day")

global_investment = pd.read_csv(
    "/home/nox/coding/data_analytics_team_project/Global private investment in generative AI - Sheet1.csv"
)

moores_law = pd.read_csv(
    "/home/nox/coding/data_analytics_team_project/transistors-per-microprocessor.csv"
)

semiconductor_ppi = pd.read_csv(
    "/home/nox/coding/data_analytics_team_project/PCU3344133344134.csv"
)


def training_scatter():
    """scatter plot of training compute"""
    sns.scatterplot(
        x="Entity", y="Training computation (petaFLOP)", data=training_compute
    )
    plt.show()


def training_relplot():
    """replot of training compute"""
    g = sns.relplot(
        x="Entity", y="Training computation (petaFLOP)", data=training_compute
    )
    g.set(xscale="log", yscale="log")
    plt.show()


def training_regplot():
    """regplot of training compute"""
    tc = training_compute.copy()

    tc["Day"] = (
        pd.to_datetime(tc["Day"], infer_datetime_format=True).astype("int64") // 10**9
    )
    start_time = datetime.strptime(START_DATE, "%Y-%m-%d").timestamp()
    tc = tc[tc["Day"] >= start_time]

    tc["log_compute"] = np.log10(tc["Training computation (petaFLOP)"])

    sns.regplot(x="Day", y="log_compute", data=tc, scatter=True)

    ax = plt.gca()
    yticks = ax.get_yticks()
    ax.set_yticklabels([f"$10^{{{v:.0f}}}$" for v in yticks])
    plt.ylabel("Training computation (petaFLOP)")
    plt.xlabel("Unix Timestamp")
    plt.show()

def training_linregress():
    tc = training_compute.copy()
    tc["Day"] = (
        pd.to_datetime(tc["Day"], infer_datetime_format=True).astype("int64") // 10**9
    )
    start_time = datetime.strptime(START_DATE, "%Y-%m-%d").timestamp()
    tc = tc[tc["Day"] >= start_time]

    slope, intercept, r_value, p_value, std_err = stats.linregress(tc["Day"].tolist(), tc["Training computation (petaFLOP)"].tolist())

    return (slope, intercept)

def global_investment_bar():
    sns.barplot(x="Year", y="Total investment (in billions)", data=global_investment)
    plt.show()

def moores_law_regplot():
    ml = moores_law.copy()
    ml["log_transistors"] = np.log10(ml["Transistors per microprocessor"])

    sns.regplot(x="Year",y="log_transistors",data=ml)

    ax = plt.gca()
    yticks = ax.get_yticks()
    ax.set_yticklabels([f"$10^{{{v:.0f}}}$" for v in yticks])
    plt.ylabel("Transistors per microprocessor")
    plt.show()

def semiconductor_ppi_linregress():
    sppi = semiconductor_ppi.copy()
    sppi = sppi.dropna()
    sppi["observation_date"] = (
        pd.to_datetime(sppi["observation_date"], infer_datetime_format=True).astype("int64") // 10**9
    )

    slope, intercept, r_value, p_value, std_err = stats.linregress(sppi["observation_date"].tolist(), sppi["PCU3344133344134"].tolist())

    return (slope, intercept)

def semiconductor_ppi_regplot():
    sppi = semiconductor_ppi.copy()
    sppi["observation_date"] = (
        pd.to_datetime(sppi["observation_date"], infer_datetime_format=True).astype("int64") // 10**9
    )

    sns.regplot(x="observation_date", y="PCU3344133344134", data=sppi)

    plt.ylabel("Index Jun 1981=100")
    plt.xlabel("Unix Timestamp")
    plt.show()


print(training_linregress())
# training_regplot()
# global_investment_bar()
# moores_law_regplot()
# print(semiconductor_ppi_linregress())
# semiconductor_ppi_regplot()
