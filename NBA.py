from flask import Flask, redirect, render_template, request, url_for
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


class NBA:
    def __init__(self, results, team_stats, advanced_stats):
        self.results = pd.read_csv(results)
        # Get unique teams
        self.teams = self.results["WinTeam"].unique()
        # self.ELOs: team name => ELO score
        self.ELOs = {}
        self.team_stats = pd.read_csv(team_stats)
        self.advanced_stats = pd.read_csv(advanced_stats)
        # Merge "2020_2021_TeamStats.csv" and "2020_2021_AdvancedStats.csv" together on column "Team"
        self.stats = pd.merge(self.team_stats, self.advanced_stats, how="left", on="Team").set_index("Team")
        # Features dataset with dimension 1171(#matches) * 92(#features)
        self.X = []
        # Match results dataset with dimension 1171(#matches) * 1
        self.y = []
        # Use Logistic Regression as model
        self.model = LogisticRegression(max_iter=2000)
        # Set up self.X
        self.build_dataset()
        self.model.fit(self.X, self.y)
        # Accuracy is around 65.5% on average
        print(cross_val_score(self.model, self.X, self.y, cv=10, scoring="accuracy", n_jobs=-1).mean())

    def build_dataset(self):
        # Initialize each team's elo to 1600
        for team in self.teams:
            self.ELOs[team] = 1600
        # Build dataset
        for index, row in self.results.iterrows():
            win_team, lose_team, win_loc = row["WinTeam"], row["LoseTeam"], row["WinLoc"]
            # Add 100 elo to the home team
            win_team_elo = self.ELOs[win_team] + 70 if win_loc == "H" else self.ELOs[win_team]
            lose_team_elo = self.ELOs[lose_team] + 70 if win_loc == "V" else self.ELOs[lose_team]
            # First feature is ELO score
            win_team_features = [win_team_elo]
            # Add other features
            for key, value in self.stats.loc[win_team].iteritems():
                win_team_features.append(value)
            win_team_features = np.nan_to_num(win_team_features)
            # Same for lose team
            lose_team_features = [lose_team_elo]
            for key, value in self.stats.loc[lose_team].iteritems():
                lose_team_features.append(value)
            lose_team_features = np.nan_to_num(lose_team_features)
            # Randomly add features and label of win team or lose team
            if random.random() > 0.5:
                self.X.append(np.append(win_team_features, lose_team_features))
                self.y.append(1)
            else:
                self.X.append(np.append(lose_team_features, win_team_features))
                self.y.append(0)
            # Update ELO score based on match result
            self.update_elo(win_team, lose_team)

        # Sort ELOs based on value descending for generate_elo_chart
        self.ELOs = dict(sorted(self.ELOs.items(), key=lambda x: x[1]))

    # Update ELO based on matches results
    def update_elo(self, win_team, lose_team):
        # Use the exact same formula given in "introduction.html"
        win_team_elo = self.ELOs[win_team]
        lose_team_elo = self.ELOs[lose_team]
        win_team_exp = (lose_team_elo - win_team_elo) / 400
        lose_team_exp = (win_team_elo - lose_team_elo) / 400
        # The expected probability of win_team wins and lose_team loses
        win_to_lose_E = 1 / (1 + 10 ** win_team_exp)
        # The expected probability of win_team wins and lose_team loses
        lose_to_win_E = 1 / (1 + 10 ** lose_team_exp)
        self.ELOs[win_team] = round(win_team_elo + 32 * (1 - win_to_lose_E))
        self.ELOs[lose_team] = round(lose_team_elo + 32 * -lose_to_win_E)

    # Predict the match result between two teams
    def predict(self, visit_team, home_team):
        # First feature is ELO score
        features = [self.ELOs[visit_team]]
        for key, value in self.stats.loc[visit_team].iteritems():
            features.append(value)
        # Home team get an advantage of 70 ELO score
        features.append(self.ELOs[home_team] + 70)
        for key, value in self.stats.loc[home_team].iteritems():
            features.append(value)
        features = np.nan_to_num(features)
        return self.model.predict_proba([features])[0][0]

    # Generate ELO chart
    def generate_elo_chart(self):
        fig, ax = plt.subplots()
        keys = list(self.ELOs.keys())
        values = list(self.ELOs.values())
        # Use a horizontal bar chart
        bars = plt.barh(range(len(values)), sorted(values), tick_label=keys, color="moccasin", height=0.6)
        # Image settings
        plt.bar_label(bars, color="green")
        plt.title("NBA 2020-2021 Season Teams ELO Score")
        plt.xlabel("ELO Score")
        plt.ylabel("Team")
        fig.set_size_inches(15, 5)
        plt.tight_layout()
        plt.savefig("static/elo.png", dpi=100)

    # Generate prediction chart
    def generate_prediction_chart(self, visit_team, home_team):
        # If two teams are the same, they have 50% winning chance each (If not preprocessing there's a home advantage)
        home_team_win_rate = self.predict(visit_team, home_team) if visit_team != home_team else 0.5
        visit_team_win_rate = 1 - home_team_win_rate if visit_team != home_team else 0.5
        labels = ["Visit: " + visit_team, "Home: " + home_team]
        sizes = [100 * visit_team_win_rate, 100 * home_team_win_rate]
        fig, ax = plt.subplots()
        # Use a pie chart
        ax.pie(sizes, autopct="%1.1f%%")
        # Image settings
        ax.axis("equal")
        ax.legend(labels, fontsize=8)
        plt.title(visit_team + "  VS.  " + home_team)
        plt.savefig("static/prediction.png", dpi=100)


# NBA model and Flask website
NBA = NBA("2020_2021_results.csv", "2020_2021_TeamStats.csv", "2020_2021_AdvancedStats.csv")
app = Flask(__name__)


# Redirect user to "introduction.html"
@app.route("/")
def home():
    return redirect(url_for("introduction"))


@app.route("/introduction")
def introduction():
    return render_template("introduction.html")


# Show each team's ELO score
@app.route("/showelo")
def showelo():
    NBA.generate_elo_chart()
    # teams is used for the selection of two teams to predict
    return render_template("showelo.html", teams=NBA.teams)


# Show the predicted win rates of two teams
@app.route("/showprediction", methods=["GET", "POST"])
def showprediction():
    if request.method == "POST":
        visit_team = request.form["visit_team"]
        home_team = request.form["home_team"]
        NBA.generate_prediction_chart(visit_team, home_team)
        return render_template("showprediction.html")
    # If user directly goes into this url, display an error message
    else:
        return render_template("noteamselected.html")


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)
