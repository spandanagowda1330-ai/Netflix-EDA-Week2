"""
Exploratory Data Analysis - Netflix Titles Dataset
Week 2 Internship Task
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

BASE = "."
FIGS = f"{BASE}/figs"
os.makedirs(FIGS, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 130

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("netflix_titles.csv")

print("="*70)
print("SHAPE:", df.shape)
print("="*70)
print(df.dtypes)
print("="*70)
print(df.describe(include="all").T)

# ---------------------------------------------------------------
# 2. MISSING VALUES
# ---------------------------------------------------------------
missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
missing_table = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_table = missing_table[missing_table.missing_count > 0]
print("="*70)
print("MISSING VALUES")
print(missing_table)
missing_table.to_csv(f"missing_values.csv")

# ---------------------------------------------------------------
# 3. DATA CLEANING / TRANSFORMATION
# ---------------------------------------------------------------
df_clean = df.copy()

# Fill categorical missing with explicit placeholder (do NOT drop rows - too much loss)
for col in ["director", "cast", "country"]:
    df_clean[col] = df_clean[col].fillna("Unknown")
df_clean["rating"] = df_clean["rating"].fillna(df_clean["rating"].mode()[0])
df_clean = df_clean.dropna(subset=["date_added"])  # small number of rows, needed for date features

# Parse date_added
df_clean["date_added"] = pd.to_datetime(df_clean["date_added"].str.strip(), format="mixed")
df_clean["year_added"] = df_clean["date_added"].dt.year
df_clean["month_added"] = df_clean["date_added"].dt.month_name()

# Primary country only (first listed)
df_clean["primary_country"] = df_clean["country"].apply(lambda x: x.split(",")[0].strip())

# Duration split by type
def parse_duration(row):
    d = str(row["duration"])
    if row["type"] == "Movie" and "min" in d:
        return int(d.replace(" min", ""))
    return np.nan

def parse_seasons(row):
    d = str(row["duration"])
    if row["type"] == "TV Show" and "Season" in d:
        return int(d.split(" ")[0])
    return np.nan

df_clean["duration_minutes"] = df_clean.apply(parse_duration, axis=1)
df_clean["num_seasons"] = df_clean.apply(parse_seasons, axis=1)

# Age of content when added to Netflix
df_clean["content_age_at_add"] = df_clean["year_added"] - df_clean["release_year"]

df_clean.to_csv("netflix_clean.csv", index=False)
print("="*70)
print("Post-cleaning shape:", df_clean.shape)

# ---------------------------------------------------------------
# 4. KEY SUMMARY STATS
# ---------------------------------------------------------------
type_counts = df_clean["type"].value_counts()
print("Type counts:\n", type_counts)
print("Movie duration stats:\n", df_clean["duration_minutes"].describe())
print("TV seasons stats:\n", df_clean["num_seasons"].describe())

# Genre explosion (listed_in is comma separated)
genre_series = df_clean["listed_in"].str.split(", ").explode()
top_genres = genre_series.value_counts().head(15)
print("Top genres:\n", top_genres)

# Country explosion
country_series = df_clean["country"].str.split(", ").explode()
country_series = country_series[country_series != "Unknown"]
top_countries = country_series.value_counts().head(10)
print("Top countries:\n", top_countries)

# =================================================================
# VISUALIZATIONS
# =================================================================

# Fig 1: Movies vs TV Shows
plt.figure(figsize=(6, 4.5))
colors = ["#E50914", "#221f1f"]
ax = type_counts.plot(kind="bar", color=colors)
plt.title("Content Type: Movies vs TV Shows", fontsize=12, fontweight="bold")
plt.xlabel("Content Type")
plt.ylabel("Number of Titles")
plt.xticks(rotation=0)
for i, v in enumerate(type_counts):
    ax.text(i, v + 30, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGS}/01_type_counts.png")
plt.close()

# Fig 2: Titles added per year
plt.figure(figsize=(8, 4.5))
yearly = df_clean[df_clean["year_added"] >= 2008]["year_added"].value_counts().sort_index()
ax = sns.lineplot(x=yearly.index, y=yearly.values, marker="o", color="#E50914", linewidth=2.5)
plt.title("Number of Titles Added to Netflix per Year", fontsize=12, fontweight="bold")
plt.xlabel("Year Added")
plt.ylabel("Number of Titles Added")
plt.tight_layout()
plt.savefig(f"{FIGS}/02_titles_per_year.png")
plt.close()

# Fig 3: Movie duration distribution
plt.figure(figsize=(7, 4.5))
sns.histplot(df_clean["duration_minutes"].dropna(), bins=40, color="#E50914", kde=True)
plt.title("Distribution of Movie Duration", fontsize=12, fontweight="bold")
plt.xlabel("Duration (minutes)")
plt.ylabel("Number of Movies")
plt.tight_layout()
plt.savefig(f"{FIGS}/03_movie_duration.png")
plt.close()

# Fig 4: TV show seasons distribution
plt.figure(figsize=(7, 4.5))
season_counts = df_clean["num_seasons"].value_counts().sort_index()
ax = sns.barplot(x=season_counts.index.astype(int), y=season_counts.values, color="#221f1f")
plt.title("Distribution of TV Show Season Counts", fontsize=12, fontweight="bold")
plt.xlabel("Number of Seasons")
plt.ylabel("Number of TV Shows")
plt.tight_layout()
plt.savefig(f"{FIGS}/04_tv_seasons.png")
plt.close()

# Fig 5: Top 10 genres
plt.figure(figsize=(8, 5))
top10_genres = top_genres.head(10).sort_values()
plt.barh(top10_genres.index, top10_genres.values, color="#E50914")
plt.title("Top 10 Most Common Genres", fontsize=12, fontweight="bold")
plt.xlabel("Number of Titles")
plt.ylabel("Genre")
plt.tight_layout()
plt.savefig(f"{FIGS}/05_top_genres.png")
plt.close()

# Fig 6: Top 10 countries
plt.figure(figsize=(8, 5))
top10_countries = top_countries.head(10).sort_values()
plt.barh(top10_countries.index, top10_countries.values, color="#221f1f")
plt.title("Top 10 Content-Producing Countries", fontsize=12, fontweight="bold")
plt.xlabel("Number of Titles")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig(f"{FIGS}/06_top_countries.png")
plt.close()

# Fig 7: Content ratings distribution, split by type
plt.figure(figsize=(9, 5))
top_ratings = df_clean["rating"].value_counts().head(8).index
rating_type = df_clean[df_clean["rating"].isin(top_ratings)]
ax = sns.countplot(data=rating_type, y="rating", hue="type", order=top_ratings,
                    palette=["#E50914", "#221f1f"])
plt.title("Content Rating Distribution by Type", fontsize=12, fontweight="bold")
plt.xlabel("Number of Titles")
plt.ylabel("Rating")
plt.legend(title="Type")
plt.tight_layout()
plt.savefig(f"{FIGS}/07_ratings_by_type.png")
plt.close()

# Fig 8: Release year trend by type (content production over time, not addition)
plt.figure(figsize=(9, 5))
recent = df_clean[df_clean["release_year"] >= 1990]
sns.histplot(data=recent, x="release_year", hue="type", multiple="stack",
             bins=35, palette=["#E50914", "#221f1f"])
plt.title("Release Year Distribution by Content Type", fontsize=12, fontweight="bold")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.savefig(f"{FIGS}/08_release_year_by_type.png")
plt.close()

# Fig 9: Content age at time of adding to Netflix (movies vs TV)
plt.figure(figsize=(7.5, 4.5))
plot_df = df_clean[(df_clean["content_age_at_add"] >= 0) & (df_clean["content_age_at_add"] <= 30)]
sns.boxplot(data=plot_df, x="type", y="content_age_at_add", palette=["#E50914", "#221f1f"])
plt.title("Content Age When Added to Netflix, by Type", fontsize=12, fontweight="bold")
plt.xlabel("Content Type")
plt.ylabel("Years Between Release and Adding to Netflix")
plt.tight_layout()
plt.savefig(f"{FIGS}/09_content_age_at_add.png")
plt.close()

# Fig 10: Monthly addition seasonality heatmap (year x month)
plt.figure(figsize=(10, 5.5))
month_order = ["January","February","March","April","May","June","July",
               "August","September","October","November","December"]
heat_data = df_clean[df_clean["year_added"] >= 2015].groupby(
    ["year_added", "month_added"]).size().unstack(fill_value=0)
heat_data = heat_data.reindex(columns=month_order)
sns.heatmap(heat_data, cmap="Reds", annot=True, fmt="d", cbar_kws={"label": "Titles Added"})
plt.title("Titles Added by Month and Year (Seasonality)", fontsize=12, fontweight="bold")
plt.xlabel("Month")
plt.ylabel("Year")
plt.tight_layout()
plt.savefig(f"{FIGS}/10_seasonality_heatmap.png")
plt.close()

print("="*70)
print("All figures saved to", FIGS)
print(sorted(os.listdir(FIGS)))
