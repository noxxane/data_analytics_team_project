"""data analytics team project analyzing ai data"""

import platform
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

START_DATE = "1970-1-2"

os_name = platform.system()
if os_name == "Windows":
    project_dir = Path(
        "C:/Users/Colie/OneDrive/Desktop/coding/data_analytics_team_project"
    )
else:
    project_dir = Path("/home/nox/coding/data_analytics_team_project/")

training_compute_path = project_dir / "artificial-intelligence-training-computation.csv"
training_compute = pd.read_csv(training_compute_path)
training_compute = training_compute.sort_values(by="Day")

global_investment_path = (
    project_dir / "Global private investment in generative AI - Sheet1.csv"
)
global_investment = pd.read_csv(global_investment_path)

moores_law_path = project_dir / "transistors-per-microprocessor.csv"
moores_law = pd.read_csv(moores_law_path)

semiconductor_ppi_path = project_dir / "PCU3344133344134.csv"
semiconductor_ppi = pd.read_csv(semiconductor_ppi_path)


def training_scatter():
    """scatter plot of training compute"""
    _ = sns.scatterplot(
        x="Entity", y="Training computation (petaFLOP)", data=training_compute
    )
    plt.show()


def training_relplot():
    """replot of training compute"""
    g = sns.relplot(
        x="Entity", y="Training computation (petaFLOP)", data=training_compute
    )
    _ = g.set(xscale="log", yscale="log")
    plt.show()


def timestamp_to_date(x, pos):
    return datetime.utcfromtimestamp(x).strftime("%Y-%m-%d")


def training_regplot():
    """regplot of training compute"""
    first_start_date = "1971-1-1"
    second_start_date = "2011-1-1"
    start_dates = [first_start_date, second_start_date]
    for start_date in start_dates:
        tc = training_compute.copy()

        tc["Day"] = (
            pd.to_datetime(tc["Day"], infer_datetime_format=True).astype("int64")
            // 10**9
        )
        start_time = datetime.strptime(start_date, "%Y-%m-%d").timestamp()
        tc = tc[tc["Day"] >= start_time]

        tc["log_compute"] = np.log10(tc["Training computation (petaFLOP)"])

        _ = sns.regplot(x="Day", y="log_compute", data=tc, scatter=True)

        ax = plt.gca()
        yticks = ax.get_yticks()
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(timestamp_to_date))
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        ax.set_yticklabels([f"$10^{{{v:.0f}}}$" for v in yticks])
        plt.ylabel("Training computation (petaFLOP)")
        plt.xlabel("Date")
        plt.show()


def training_linregress():
    """linear regression for training compute"""
    tc = training_compute.copy()
    tc["Day"] = (
        pd.to_datetime(tc["Day"], infer_datetime_format=True).astype("int64") // 10**9
    )
    start_time = datetime.strptime(START_DATE, "%Y-%m-%d").timestamp()
    tc = tc[tc["Day"] >= start_time]

    slope, intercept, _, _, _ = stats.linregress(
        tc["Day"].tolist(), tc["Training computation (petaFLOP)"].tolist()
    )

    return (slope, intercept)


def exponential_model(x, a, b):
    return a * np.exp(b * x)


def global_investment_bar():
    x = np.array(global_investment["Year"].tolist())
    y = np.array(global_investment["Total investment (in billions)"].tolist())

    x_indexed = np.arange(len(x))

    params, _ = curve_fit(exponential_model, x, y, p0=[1, 0.5])
    a, b = params

    x_fit_exp = np.linspace(x_indexed.min(), x_indexed.max(), 300)
    y_fit_exp = exponential_model(x_fit_exp, a, b)

    coeffs = np.polyfit(x_indexed, y, deg=2)
    poly = np.poly1d(coeffs)

    x_fit_poly = np.linspace(x_indexed.min(), x_indexed.max(), 300)
    y_fit_poly = poly(x_fit_poly)

    _, ax = plt.subplots()
    sns.barplot(x=x, y=y, ax=ax, color="steelblue", alpha=0.6)
    ax.plot(
        x_fit_exp,
        y_fit_exp,
        color="crimson",
        linewidth=2.5,
        label=f"Exp fit: y = {a:.2f} * e^({b:.2f}x)",
    )
    ax.plot(
        x_fit_poly,
        y_fit_poly,
        color="darkorange",
        linewidth=2.5,
        label=f"Poly fit(deg=2): y = {coeffs[0]:.2f}x^2 + {coeffs[1]:.2f}x + {coeffs[2]:.2f}",
    )

    ax.legend()
    plt.tight_layout()
    plt.show()


def global_investment_bar_poly():
    x = np.array(global_investment["Year"].tolist())
    y = np.array(global_investment["Total investment (in billions)"].tolist())

    x_indexed = np.arange(len(x))

    coeffs = np.polyfit(x_indexed, y, deg=2)
    poly = np.poly1d(coeffs)

    x_fit = np.linspace(x_indexed.min(), x_indexed.max(), 300)
    y_fit = poly(x_fit)


def moores_law_regplot():
    """regplot for moores law"""
    ml = moores_law.copy()
    ml["log_transistors"] = np.log10(ml["Transistors per microprocessor"])

    _ = sns.regplot(x="Year", y="log_transistors", data=ml)

    ax = plt.gca()
    yticks = ax.get_yticks()
    ax.set_yticklabels([f"$10^{{{v:.0f}}}$" for v in yticks])
    plt.ylabel("Transistors per microprocessor")
    plt.show()


def semiconductor_ppi_linregress():
    """linear regression for semiconductor ppi"""
    sppi = semiconductor_ppi.copy()
    sppi = sppi.dropna()
    sppi["observation_date"] = (
        pd.to_datetime(sppi["observation_date"], infer_datetime_format=True).astype(
            "int64"
        )
        // 10**9
    )

    slope, intercept, _, _, _ = stats.linregress(
        sppi["observation_date"].tolist(), sppi["PCU3344133344134"].tolist()
    )

    return (slope, intercept)


def semiconductor_ppi_regplot():
    """regplot for semiconductor ppi"""
    sppi = semiconductor_ppi.copy()
    sppi["observation_date"] = (
        pd.to_datetime(sppi["observation_date"], infer_datetime_format=True).astype(
            "int64"
        )
        // 10**9
    )

    ax = sns.regplot(x="observation_date", y="PCU3344133344134", data=sppi)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(timestamp_to_date))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.ylabel("Index Jun 1981=100")
    plt.xlabel("Date")
    plt.show()


print(training_linregress())
training_regplot()
global_investment_bar()
moores_law_regplot()
print(semiconductor_ppi_linregress())
semiconductor_ppi_regplot()
