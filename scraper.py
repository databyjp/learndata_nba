# ========== (c) JP Hwang 22/8/20  ==========

import logging

# ===== START LOGGER =====
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
root_logger.addHandler(sh)

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment


"""
https://www.basketball-reference.com/leagues/NBA_2020_totals.html
https://www.basketball-reference.com/leagues/NBA_2020_per_game.html
https://www.basketball-reference.com/leagues/NBA_2020_shooting.html
"""


def scrape_bballref_players(year=2020, stat="totals", pause=2):

    import time
    time.sleep(pause)

    if stat == "pergame":
        div_id = "div_per_game_stats"
        start_page = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
        header_row = 0
    elif stat == "shooting":
        div_id = "div_shooting_stats"
        start_page = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_shooting.html"
        header_row = 1
    elif stat == "totals":
        div_id = "div_totals_stats"
        start_page = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_totals.html"
        header_row = 0
    elif stat == "advanced":
        div_id = "div_advanced_stats"
        start_page = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_advanced.html"
        header_row = 0

    logger.info(f"Getting {stat} stats for {year}...")
    page = requests.get(start_page)
    soup = BeautifulSoup(page.text, "html.parser")

    div = soup.find(id=div_id)
    comments = div.find_all(text=lambda text: isinstance(text, Comment))  # find hidden table
    if len(comments) > 0:
        div = BeautifulSoup(comments[0].extract(), "html.parser")

    # ========== PARSE EACH ROW OF DATA ==========
    data_rows = div.find_all("tr")  # Includes the header row!
    parsed_data = list()
    player_attr = {"data-stat": "player"}
    stat_keys = [col.attrs["data-stat"] for col in data_rows[header_row].find_all("th")]
    # stat_names = [col.attrs["aria-label"] for col in data_rows[0].find_all("th")]

    for row in data_rows:
        tmp_data = dict()
        if row.find(attrs=player_attr) is not None:
            if row.find(attrs=player_attr).find("a") is not None:
                player_link = row.find(attrs=player_attr).find("a").attrs["href"]
                tmp_data["name"] = row.find(attrs=player_attr).find("a").text
                for attr in stat_keys[2:]:
                    tmp_data[attr] = row.find(attrs={"data-stat": attr}).text
                tmp_data["link"] = player_link
                parsed_data.append(tmp_data)

    data_df = pd.DataFrame(parsed_data)

    return data_df


def scrape_team_stats(soup_in, div_id="all_team-stats-per_game"):

    div = soup_in.find(id=div_id)
    comments = div.find_all(text=lambda text: isinstance(text, Comment))  # find hidden table
    if len(comments) > 0:
        div = BeautifulSoup(comments[0].extract(), "html.parser")

    # ========== PARSE EACH ROW OF DATA ==========
    data_rows = div.find_all("tr")  # Includes the header row!
    parsed_data = list()
    team_attr = {"data-stat": "team_name"}
    stat_keys = [col.attrs["data-stat"] for col in data_rows[0].find_all("th")]
    # stat_names = [col.attrs["aria-label"] for col in data_rows[0].find_all("th")]

    for row in data_rows:
        tmp_data = dict()
        if row.find(attrs=team_attr).find("a") is not None:
            team_link = row.find(attrs=team_attr).find("a").attrs["href"]
            tmp_data["name"] = row.find(attrs=team_attr).find("a").text
            for attr in stat_keys[2:]:
                tmp_data[attr] = row.find(attrs={"data-stat": attr}).text
            tmp_data["link"] = team_link
            parsed_data.append(tmp_data)

    data_df = pd.DataFrame(parsed_data)
    data_df.drop("DUMMY", axis=1, inplace=True)

    return data_df


def scrape_dataset(stat, start_yr=1997, end_yr=2021, outfile=None):
    out_dfs = list()
    for yr in range(start_yr, end_yr):
        tmp_df = scrape_bballref_players(year=yr, stat=stat)
        tmp_df = tmp_df.assign(season=yr)
        out_dfs.append(tmp_df)
    concat_df = pd.concat(out_dfs, axis=0).reset_index()

    if outfile is not None:
        concat_df.to_csv(outfile)

    return out_dfs


# player_per_game_df = scrape_dataset("per_game", 1981, 2021, "data/player_per_game.csv")
# player_shooting_df = scrape_dataset("shooting", 1997, 2021, "data/player_shooting.csv")
# player_totals_df = scrape_dataset("totals", 1981, 2021, "data/player_totals.csv")
player_advanced_df = scrape_dataset("advanced", 1981, 2021, "data/player_advanced.csv")
