"""Database関連のモジュール."""
import os

import sqlalchemy
import sqlalchemy.ext.declarative
import sqlalchemy.orm

import src.data_path as data_path

_DEFAULT_DB_URL = (
    f"sqlite:///{data_path.PROCESSED_DIR.joinpath('db.sqlite3').as_posix()}"
)
_DB_URL = os.environ.get("DB_URL", _DEFAULT_DB_URL)

ENGINE = sqlalchemy.create_engine(_DB_URL, echo=False)
Session = sqlalchemy.orm.scoped_session(
    sqlalchemy.orm.sessionmaker(bind=ENGINE, autocommit=False, autoflush=True)
)
Base = sqlalchemy.ext.declarative.declarative_base()
