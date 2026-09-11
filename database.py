from sqlalchemy import create_engine, Column, String, Float, Date
from sqlalchemy.orm import declarative_base, sessionmaker

# Local SQLite database file
DATABASE_URL = "sqlite:///portfolio.db"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class AssetPrice(Base):
    __tablename__ = "asset_prices"

    # Composite primary key: one unique record per ticker per date
    ticker = Column(String(20), primary_key=True)
    trade_date = Column(Date, primary_key=True)
    close_price = Column(Float, nullable=False)
    daily_return = Column(Float, nullable=True)
    rolling_vol = Column(Float, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database and 'asset_prices' table created successfully.")

if __name__ == "__main__":
    init_db()