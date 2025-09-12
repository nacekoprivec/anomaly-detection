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
    status = Column(String, nullable=True, default="inactive") # e.g., active, inactive, error

    # 1 - n relationship
    logs = relationship("Log", back_populates="detector")


class Log(Base):
    __tablename__ = "logs"  

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    detector_id = Column(Integer, ForeignKey("anomaly_detectors.id"), nullable=False)

    start_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    end_at = Column(DateTime, index=True)
    config = Column(Text, nullable=False) 
    config_name = Column(String, nullable=True, index=True)

    duration_seconds = Column(Integer, nullable=True, index=True)
    # n - 1 relationship
    detector = relationship("AnomalyDetector", back_populates="logs")

Base.metadata.create_all(bind=engine)
