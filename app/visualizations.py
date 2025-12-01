import matplotlib.pyplot as plt
import seaborn as sns

def plot_attrition_by_column(df, column):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(
        data=df,
        y=column,
        hue="Attrition",
        order=df[column].value_counts().index
    )
    for container in ax.containers:
        ax.bar_label(container)
    ax.invert_yaxis()
    plt.title(f"Attrition by {column}", fontsize=14)
    plt.xlabel("Count", fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.tight_layout()
    return plt

def plot_percentage_stacked(df, column):
    percent_df = (
        df.groupby([column, "Attrition"])
        .size()
        .groupby(level=0)
        .apply(lambda x: 100 * x / x.sum())
        .unstack()
        .fillna(0)
    )
    ax = percent_df.plot(
        kind="barh",
        stacked=True,
        figsize=(10, 6),
    )
    uniquelabel = []
    for label in ax.get_yticklabels():
        new_label = label.get_text().split(',')[0]
        new_label = new_label.replace("(", "", 1)
        uniquelabel.append(new_label)
    ax.set_yticklabels(uniquelabel)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    return plt
