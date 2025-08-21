from ..database import Base, engine
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship

class Log(Base):
    __tablename__ = "logs"  

    id = Column(Integer, primary_key=True, autoincrement=True)
    start_timedate = Column(DateTime, index=True, nullable=False)
    end_timedate = Column(DateTime, index=True)
    config = Column(Text, nullable=False) 

    anomalies = relationship("Anomaly", back_populates="log", cascade="all, delete-orphan")


class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(Float, nullable=False)
    ftr_vector = Column(Float, nullable=False)

    log_id = Column(Integer, ForeignKey("logs.id"), nullable=False)
    log = relationship("Log", back_populates="anomalies")


Base.metadata.create_all(bind=engine)
