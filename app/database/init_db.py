from app.database.connection import Base, engine
from app.database.models import Message


def initialize_database() -> None:
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully.")