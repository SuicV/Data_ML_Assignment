from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./database/sqlitedb.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

class SQLiteConnector:
    __instance = None
    __sqlite_db_connection = 'sqlite:///./database/sqlitedb.db'
    @staticmethod
    def get_instance():
        Static access method for singleton object
        if SQLiteConnector.__instance is None:
            SQLiteConnector()
        return SQLiteConnector.__instance

    def __init__(self):
        Virtually private constructor
        if SQLiteConnector.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            SQLiteConnector.__instance = self
            self.engine = create_engine(SQLiteConnector.__sqlite_db_connection, connect_args={"check_same_thread": False})
            self.Session = sessionmaker(bind=self.engine)
            self.Base = declarative_base()
            # check if table exists
            if not self.engine.dialect.has_table(self.engine, 'inference'):
                # create table
                self.Base.metadata.create_all(self.engine)
            

    def get_session(self):
        Returns a new session
        return self.Session

    def get_base(self):
        Returns the base
        return self.Base
"""