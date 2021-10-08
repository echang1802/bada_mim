import requests
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression

class fpl:

    def __init__(self):
        self._base_url = " https://fantasy.premierleague.com/api/"

        info = self._api_call(f"{self._base_url}bootstrap-static/")

        self._players = pd.DataFrame(info['elements']).set_index("id")
        self._elements_type = pd.DataFrame(info['element_types']).set_index("id")
        self._clubs = pd.DataFrame(info['teams']).set_index("id")

        with open("internationals.xlsx", "rb") as file:
            internationals_players = pd.read_excel(file, sheet_name = "players")
            self._get_internationals_players(internationals_players)

        self._get_history()

    def _get_internationals_players(self, players):
        self._internationals_players = self._players[["web_name", "team", "element_type"]].reset_index(drop = False)
        self._internationals_players.rename(columns = {"id": "player_id"}, inplace = True)
        self._internationals_players = self._internationals_players.set_index("team").join(self._clubs["name"], how = "left").reset_index(drop = True)
        self._internationals_players.rename(columns = {"name": "team"}, inplace = True)
        self._internationals_players = self._internationals_players.set_index("element_type").join(self._elements_type["singular_name"], how = "left").reset_index(drop = True)
        self._internationals_players.rename(columns = {"singular_name": "position"}, inplace = True)

        self._internationals_players = self._internationals_players.merge(players, on = ["team", "web_name"], how = "left")
        self._internationals_players.international.fillna(False, inplace = True)
        self._internationals_players.played.fillna(False, inplace = True)
        self._internationals_players.set_index("player_id", inplace = True)


    def _api_call(self, url):
        r = requests.get(url)
        json = r.json()
        return json

    def get_fixtures(self):
        url = f"{self._base_url}fixtures/"
        return self._api_call(url)

    def get_teams(self):
        return self._clubs

    def get_players(self):
        return self._players

    def _get_history(self):
        self._players_history = {}
        for player_id, player in self._internationals_players.iterrows():
            url = f"{self._base_url}element-summary/{player_id}/"
            info = self._api_call(url)

            if type(info) != dict or not "history" in info.keys():
                continue

            self._players_history[f"{player_id}"] = {
                "player_info" : player,
                "history" : pd.DataFrame(info['history'])
            }

    def DID(self, played = False):
        data = pd.DataFrame()
        for internationals in [True, False]:
            data = data.append(self._get_did_data_df(internationals, played), ignore_index = True)


        data = pd.melt(data, id_vars = ["international"], value_vars = [f"{i}" for i in range(1,7)])
        data.variable = data.variable.astype(float)

        did = data.groupby("variable").agg({
            "value": lambda x: x.values[0] - x.values[1]
        }).reset_index(drop = False)
        print("Diferencia antes:", did.loc[did.variable < 4].value.mean())
        print("Diferencia despues:", did.loc[did.variable > 3].value.mean())
        print("DID:", did.loc[did.variable > 3].value.mean() - did.loc[did.variable < 4].value.mean())

        data = data.append(pd.DataFrame({
            "international" : [True, False],
            "variable" : [3.5, 3.5],
            "value" : [data.loc[(data.international == 1) & (data.variable.isin([3, 4]))].value.mean(),
                       data.loc[(data.international == 0) & (data.variable.isin([3, 4]))].value.mean()]
        }))
        data["international"] = np.where(data.international, "Internationals", "Nationals")


        plot  = sns.lineplot(x = data.variable, y = data.value, hue = data.international)
        plot  = sns.lineplot(x = [3.5, 3.5], y = [1.5, 2.7], color = "red")
        plot.set(xlabel = "Matchweek",ylabel = "Avg. Players Points")
        plot.set_xticks(range(1,7))


    def _get_did_data_df(self, international, played):
        df = pd.DataFrame()
        players = self._internationals_players.loc[self._internationals_players.international == international]
        if played and international:
            players = players.loc[players.played]
        for player_id, player in players.iterrows():
            if self._players_history[str(player_id)]["history"].shape[0] < 3:
                continue

            aux = self._players_history[str(player_id)]["history"].loc[self._players_history[str(player_id)]["history"].kickoff_time < "2021-09-30T00:00:00Z"].total_points

            if aux.sum() == 0:
                continue

            df = df.append(aux)

        df.columns = ["1", "2", "3", "4", "5", "6"]
        df = df.mean()
        df["international"] = international
        return df

    def matching(self, tol = 0.005):
        data = self._propensity_score()

        internationals = data.loc[data.international].set_index("player_id")
        nationals = data.loc[~data.international].set_index("player_id")

        matchs = {"int_points_diff" : [], "nat_points_diff": [], "ps_diff" : []}
        for player_id, player in internationals.iterrows():
            nationals["ps_comparison"] = (nationals.propensity_score - player.propensity_score).abs()
            ps_diff = nationals.ps_comparison.min()
            if ps_diff > tol:
                continue
            nat_player_id = nationals.ps_comparison.idxmin()

            aux = self._players_history[str(player_id)]["history"].loc[self._players_history[str(player_id)]["history"].kickoff_time < "2021-09-30T00:00:00Z"]
            prev_int_points = aux.loc[aux.kickoff_time < "2021-09-02T00:00:00Z"].total_points.mean()
            post_int_points = aux.loc[aux.kickoff_time > "2021-09-02T00:00:00Z"].total_points.mean()
            aux = self._players_history[str(nat_player_id)]["history"].loc[self._players_history[str(nat_player_id)]["history"].kickoff_time < "2021-09-30T00:00:00Z"]
            prev_nat_points = aux.loc[aux.kickoff_time < "2021-09-02T00:00:00Z"].total_points.mean()
            post_nat_points = aux.loc[aux.kickoff_time > "2021-09-02T00:00:00Z"].total_points.mean()

            if prev_int_points != prev_int_points or post_int_points != post_int_points or prev_nat_points != prev_nat_points or post_nat_points != post_nat_points:
                continue

            matchs["int_points_diff"].append(post_int_points - prev_int_points)
            matchs["nat_points_diff"].append(post_nat_points - prev_nat_points)
            matchs["ps_diff"].append(ps_diff)

            nationals.drop(nat_player_id, inplace = True)

        print(stats.ttest_ind(matchs["int_points_diff"], matchs["nat_points_diff"]))

        N = len(matchs["nat_points_diff"])
        g = sns.violinplot(y = matchs["int_points_diff"] + matchs["nat_points_diff"], x = (["Internationals"] * N) + (["Nationals"] * N))
        g.set(title = "Distribution of total points difference")
        return

    def _propensity_score(self, played = False):
        data = pd.DataFrame()
        for internationals in [True, False]:
            data = data.append(self._get_ps_data_df(internationals, played), ignore_index = True)

        ps_model = LogisticRegression(random_state=0).fit(data.drop(columns = ["player_id", "international"]), data.international)
        data["propensity_score"] = ps_model.predict_proba(data.drop(columns = ["player_id", "international"]))[:, 0]

        return data

    def _get_ps_data_df(self, international, played):
        df = pd.DataFrame()
        players = self._internationals_players.loc[self._internationals_players.international == international]
        if played and international:
            players = players.loc[players.played]
        for player_id, player in players.iterrows():

            aux = self._players_history[str(player_id)]["history"].loc[self._players_history[str(player_id)]["history"].kickoff_time < "2021-09-30T00:00:00Z"]

            if aux.minutes.sum() == 0:
                continue

            aux = aux.groupby("element").agg({
                "total_points": sum, "minutes": sum, "goals_scored": sum, "assists": sum, "clean_sheets": sum, "goals_conceded": sum, "own_goals": sum,
                "yellow_cards": sum, "red_cards": sum, "bonus": sum, "influence": lambda x: np.mean(x.astype(float)),
                "creativity": lambda x: np.mean(x.astype(float)), "threat": lambda x: np.mean(x.astype(float))
            })
            aux["player_id"] = player_id
            aux["international"] = international

            df = df.append(aux)

        return df
