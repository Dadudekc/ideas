# Path: AI_Debugger_Assistant/ai_agent_project/src/agents/core/agent_base.py
# Base class for all agents

class AgentBase:
    def execute(self, task):
        raise NotImplementedError('This method should be overridden in subclasses')
