# EURO 2020, 20/21 La Liga, WorldCup 2022 dataset의 경기를 train, test DataBase에 저장
    
import warnings
from statsbombpy.api_client import NoAuthWarning
warnings.filterwarnings(action="ignore", category=NoAuthWarning, module='statsbombpy')

import tqdm
from pathlib import Path
from socceraction.data.statsbomb import StatsBombLoader
from statsbombpy import sb
import pandas as pd
from unxpass.databases import SQLiteDatabase
from sklearn.model_selection import train_test_split

def loading_DB(games, path):
    DB_PATH = Path(path)
    db = SQLiteDatabase(DB_PATH)
    games = games.reset_index()

    for index in tqdm.tqdm(range(len(games)),desc=path +" loading"):
        dataset = { "getter": "remote", "competition_id":  games.loc[index,'competition_id'],
                    "season_id" : games.loc[index,'season_id'],
                    "game_id" : games.loc[index,'game_id'] }
        try : 
            db.import_data(**dataset)
        except :
            print(dataset)

    
SBL = StatsBombLoader(getter="remote", creds={"user": None, "passwd": None})

#StatsBomb360데이터에서는 4종류의 데이터만 지원함(2022월드컵, 20남자 유에파, 22여자 유에파, 20/21 라리가)
#2022년 월드컵 경기 : 64개
#20/21  라리가 경기 : 51개
#2020년 유로파 경기 : 35개
#라리가는 이유는 모르지만 에러가 발생, 여자 유로파는 성별차이로 제외
competition = sb.competitions()
competition = competition[(competition['match_available_360'].notna()) & (competition['competition_gender']=='male')]
competition = competition[(competition.competition_id == 43) & (competition.season_id == 106) |
            (competition.competition_id == 55) & (competition.season_id == 43)]

games = pd.concat([SBL.games(row.competition_id, row.season_id) for row in competition.itertuples()])
train_games, test_games = train_test_split(games,test_size=0.2,stratify=games['competition_id'])

loading_DB(train_games,"./stores/train-database.sqlite")
loading_DB(test_games,"./stores/test-database.sqlite")

