# backend/models/user.py

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from data_store import Base

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Integer, default=1)  # 1 for active, 0 for inactive

    # Relationships
    subscriptions = relationship("Subscription", back_populates="owner")
