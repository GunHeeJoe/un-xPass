# EURO 2020, 20/21 La Liga, WorldCup 2022 dataset의 경기의
#모든 feature, label불러오서 저장하기
    
from pathlib import Path
import pandas as pd
pd.set_option('display.max_columns', None)

from unxpass.databases import SQLiteDatabase
from unxpass.datasets import PassesDataset
from unxpass import features as fs
from unxpass import labels as ls

train_DB_PATH = Path("./stores/train-database.sqlite")
train_db = SQLiteDatabase(train_DB_PATH)

test_DB_PATH = Path("./stores/test-database.sqlite")
test_db = SQLiteDatabase(test_DB_PATH)

STORES_FP = Path("./stores")

train_dataset = PassesDataset(
    path=STORES_FP / "datasets" / "default" / "train",
    xfns= [f.__name__ for f in fs.all_features],
    yfns= [f.__name__ for f in ls.all_labels]   
)

test_dataset = PassesDataset(
    path=STORES_FP / "datasets" / "default" / "test",
    xfns= [f.__name__ for f in fs.all_features],
    yfns= [f.__name__ for f in ls.all_labels]   
)

train_dataset.create(train_db)
test_dataset.create(test_db)

train_db.close()
test_db.close()