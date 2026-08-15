import ast
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

df = pd.read_csv("movies.csv")
numeric = ["budget","revenue","popularity","runtime","vote_average"]
for c in numeric:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.drop_duplicates()
df = df[(df["budget"] > 0) & (df["revenue"] > 0)]
df = df.dropna(subset=numeric).copy()
df["success"] = (df["revenue"] > df["budget"]).astype(int)

# T-test
s = df.loc[df.success == 1, "popularity"]
u = df.loc[df.success == 0, "popularity"]
t_stat, t_p = stats.ttest_ind(s, u, equal_var=False, nan_policy="omit")

# Chi-square using exploded genres
# NOTE: `genres` is stored as a stringified list of TMDB dicts, e.g.
#   "[{'id': 10749, 'name': 'Romance'}]"
# A naive comma/pipe split shreds this into fragments like "[{'id': 10749"
# and "'name': 'Romance'}]" instead of the genre name. ast.literal_eval
# parses it correctly as a real Python list of dicts.
def parse_genres(x):
    try:
        parsed = ast.literal_eval(x)
        names = [g["name"] for g in parsed]
        return names if names else ["Unknown"]
    except (ValueError, SyntaxError, TypeError):
        return ["Unknown"]

genre_df = df.assign(genre_list=df["genres"].fillna("[]").apply(parse_genres)).explode("genre_list")
table = pd.crosstab(genre_df["genre_list"], genre_df["success"])
chi2, chi_p, dof, _ = stats.chi2_contingency(table)

# Random Forest — revenue excluded to prevent target leakage
features = ["budget", "popularity", "runtime", "vote_average"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["success"], test_size=.20, random_state=42, stratify=df["success"]
)
model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Rows:", len(df))
print("Success rate:", df.success.mean())
print("T-test p-value:", t_p)
print("Chi-square p-value:", chi_p)
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, zero_division=0))
print("Recall:", recall_score(y_test, pred, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))
print("Feature importance:")
print(pd.Series(model.feature_importances_, index=features).sort_values(ascending=False))
