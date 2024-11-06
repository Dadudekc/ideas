from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from db.base import Base

class AIAgent(Base):
    __tablename__ = 'ai_agents'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)

    project = relationship("Project", back_populates="ai_agents")

    def __repr__(self):
        return f"<AIAgent(name='{self.name}', project_id={self.project_id})>"
