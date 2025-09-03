from ..database import Base, engine
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone


class AnomalyDetector(Base):
    __tablename__ = "anomaly_detectors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    status = Column(String, nullable=True, default="inactive")

    # 1 - n relationship
    logs = relationship("Log", back_populates="detector")


class Log(Base):
    __tablename__ = "logs"  

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    detector_id = Column(Integer, ForeignKey("anomaly_detectors.id"), nullable=False)

    start_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    end_at = Column(DateTime, index=True)
    config = Column(Text, nullable=False) 

    duration_seconds = Column(Integer, nullable=True, index=True)

    tp = Column(Integer, nullable=True, index=True, default=0)
    tn = Column(Integer, nullable=True, index=True, default=0)
    fp = Column(Integer, nullable=True, index=True, default=0)
    fn = Column(Integer, nullable=True, index=True, default=0)
    precision = Column(Float, nullable=True, index=True)
    recall = Column(Float, nullable=True, index=True)
    f1 = Column(Float, nullable=True, index=True)

    # n - 1 relationship
    detector = relationship("AnomalyDetector", back_populates="logs")

    # 1 - n relationship
    datapoints = relationship("DataPoint", back_populates="log", cascade="all, delete-orphan")


class DataPoint(Base):
    __tablename__ = "datapoints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(Float, nullable=False)
    ftr_vector = Column(Float, nullable=False)
    is_anomaly = Column(Integer, nullable=True, default=0)

    log_id = Column(Integer, ForeignKey("logs.id"), nullable=False)

    # n - 1 relationship
    log = relationship("Log", back_populates="datapoints")


Base.metadata.create_all(bind=engine)
