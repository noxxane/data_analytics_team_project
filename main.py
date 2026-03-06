"""data analytics team project analyzing ai data"""

from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

training_compute = pd.read_csv(
    "/home/nox/coding/data_analytics_team_project/artificial-intelligence-training-computation.csv"
)
training_compute = training_compute.sort_values(by="Day")

global_investment = pd.read_csv(
    "/home/nox/coding/data_analytics_team_project/Global private investment in generative AI - Sheet1.csv"
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
    start_time = datetime.strptime("2013-1-1", "%Y-%m-%d").timestamp()
    tc = tc[tc["Day"] >= start_time]

    tc["log_compute"] = np.log10(tc["Training computation (petaFLOP)"])

    sns.regplot(x="Day", y="log_compute", data=tc, scatter=True)

    ax = plt.gca()
    yticks = ax.get_yticks()
    ax.set_yticklabels([f"$10^{{{v:.0f}}}$" for v in yticks])
    plt.ylabel("Training computation (petaFLOP)")
    plt.show()


def global_investment_bar():
    sns.barplot(x="Year", y="Total investment (in billions)", data=global_investment)
    plt.show()


global_investment_bar()
