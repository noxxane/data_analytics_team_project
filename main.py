"""data analytics team project analyzing ai data"""

import platform
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedLocator
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
    plt.show()  # pyright: ignore[reportUnknownMemberType]


def training_relplot():
    """replot of training compute"""
    g = sns.relplot(
        x="Entity", y="Training computation (petaFLOP)", data=training_compute
    )
    _ = g.set(xscale="log", yscale="log")
    plt.show()  # pyright: ignore[reportUnknownMemberType]


def timestamp_to_date(x: float, _: int | None) -> str:
    return datetime.fromtimestamp(x, UTC).strftime("%Y-%m-%d")


def training_regplot():
    """regplot of training compute"""
    first_start_date = "1971-1-1"
    second_start_date = "2011-1-1"
    start_dates = [first_start_date, second_start_date]
    for start_date in start_dates:
        tc = training_compute.copy()

        tc["Day"] = pd.to_datetime(tc["Day"]).astype("int64") // 10**9
        start_time = datetime.strptime(start_date, "%Y-%m-%d").timestamp()
        tc = tc[tc["Day"] >= start_time]

        tc["log_compute"] = np.log10(tc["Training computation (petaFLOP)"])

        ax = sns.regplot(x="Day", y="log_compute", data=tc, scatter=True)

        yticks = ax.get_yticks()
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(timestamp_to_date))
        _ = plt.xticks(rotation=45, ha="right")  # pyright: ignore[reportUnknownMemberType]
        plt.tight_layout()

        ax.yaxis.set_major_locator(FixedLocator(yticks.tolist()))  # pyright: ignore[reportAny]
        _ = ax.set_yticklabels([f"$10^{{{v:.0f}}}$" for v in yticks])  # pyright: ignore[reportUnknownMemberType, reportAny]
        _ = plt.ylabel("Training computation (petaFLOP)")  # pyright: ignore[reportUnknownMemberType]
        _ = plt.xlabel("Date")  # pyright: ignore[reportUnknownMemberType]
        plt.show()  # pyright: ignore[reportUnknownMemberType]


def training_linregress():
    """linear regression for training compute"""
    tc = training_compute.copy()
    tc["Day"] = pd.to_datetime(tc["Day"]).astype("int64") // 10**9
    start_time = datetime.strptime(START_DATE, "%Y-%m-%d").timestamp()
    tc = tc[tc["Day"] >= start_time]

    slope, intercept, _, _, _, _ = stats.linregress(
        tc["Day"].tolist(), tc["Training computation (petaFLOP)"].tolist()
    )

    return (slope, intercept)


def exponential_model(x: np.ndarray, a: np.floating, b: np.floating) -> np.ndarray:
    return a * np.exp(b * x)


def global_investment_bar():
    x = np.array(global_investment["Year"].tolist())
    y = np.array(global_investment["Total investment (in billions)"].tolist())

    x_indexed = np.arange(len(x))

    params, _ = curve_fit(exponential_model, x, y, p0=[1, 0.5])
    a, b = params

    x_fit_exp = np.linspace(x_indexed.min(), x_indexed.max(), 300)  # pyright: ignore[reportUnknownVariableType, reportAny]
    y_fit_exp = exponential_model(x_fit_exp, a, b)  # pyright: ignore[reportUnknownArgumentType]

    coeffs = np.polyfit(x_indexed, y, deg=2)
    poly = np.poly1d(coeffs)

    x_fit_poly = np.linspace(x_indexed.min(), x_indexed.max(), 300)  # pyright: ignore[reportUnknownVariableType, reportAny]
    y_fit_poly = poly(x_fit_poly)  # pyright: ignore[reportAny, reportUnknownArgumentType]

    _, ax = plt.subplots()  # pyright: ignore[reportUnknownMemberType]
    _ = sns.barplot(x=x, y=y, ax=ax, alpha=0.6)
    _ = ax.plot(  # pyright: ignore[reportUnknownMemberType]
        x_fit_exp,  # pyright: ignore[reportUnknownArgumentType]
        y_fit_exp,
        linewidth=2.5,
        label=f"Exp fit: y = {a:.2f} * e^({b:.2f}x)",
    )
    _ = ax.plot(  # pyright: ignore[reportUnknownMemberType]
        x_fit_poly,  # pyright: ignore[reportUnknownArgumentType]
        y_fit_poly,  # pyright: ignore[reportAny]
        linewidth=2.5,
        label=f"Poly fit(deg=2): y = {coeffs[0]:.2f}x^2 + {coeffs[1]:.2f}x + {coeffs[2]:.2f}",
    )

    _ = ax.legend()  # pyright: ignore[reportUnknownMemberType]
    plt.tight_layout()
    plt.show()  # pyright: ignore[reportUnknownMemberType]


def moores_law_regplot():
    """regplot for moores law"""
    ml = moores_law.copy()
    ml["log_transistors"] = np.log10(ml["Transistors per microprocessor"])

    ax = sns.regplot(x="Year", y="log_transistors", data=ml)

    yticks = ax.get_yticks()

    ax.yaxis.set_major_locator(FixedLocator(yticks.tolist()))  # pyright: ignore[reportAny]
    _ = ax.set_yticklabels([f"$10^{{{v:.0f}}}$" for v in yticks])  # pyright: ignore[reportUnknownMemberType, reportAny]
    _ = plt.ylabel("Transistors per microprocessor")  # pyright: ignore[reportUnknownMemberType]
    plt.show()  # pyright: ignore[reportUnknownMemberType]


def semiconductor_ppi_linregress():
    """linear regression for semiconductor ppi"""
    sppi = semiconductor_ppi.copy()
    sppi = sppi.dropna()
    sppi["observation_date"] = (
        pd.to_datetime(sppi["observation_date"]).astype("int64") // 10**9
    )

    slope, intercept, _, _, _, _ = stats.linregress(
        sppi["observation_date"].tolist(), sppi["PCU3344133344134"].tolist()
    )

    return (slope, intercept)


def semiconductor_ppi_regplot():
    """regplot for semiconductor ppi"""
    sppi = semiconductor_ppi.copy()
    sppi["observation_date"] = (
        pd.to_datetime(sppi["observation_date"]).astype("int64") // 10**9
    )

    ax = sns.regplot(x="observation_date", y="PCU3344133344134", data=sppi)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(timestamp_to_date))
    _ = plt.xticks(rotation=45, ha="right")  # pyright: ignore[reportUnknownMemberType]
    plt.tight_layout()

    _ = plt.ylabel("Index Jun 1981=100")  # pyright: ignore[reportUnknownMemberType]
    _ = plt.xlabel("Date")  # pyright: ignore[reportUnknownMemberType]
    plt.show()  # pyright: ignore[reportUnknownMemberType]


print(training_linregress())
training_regplot()
global_investment_bar()
moores_law_regplot()
print(semiconductor_ppi_linregress())
semiconductor_ppi_regplot()
