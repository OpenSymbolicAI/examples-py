from datetime import date
from opensymbolicai.blueprints import PlanExecute
from opensymbolicai.core import primitive, decomposition
from opensymbolicai.llm import LLMConfig, Provider


class DateAgent(PlanExecute):
    """An agent that calculates days between dates."""

    @primitive(read_only=True)
    def today(self) -> str:
        """Get today's date in ISO format (YYYY-MM-DD)."""
        return date.today().isoformat()

    @primitive(read_only=True)
    def days_between(self, start: str, end: str) -> int:
        """Calculate days between two ISO dates (YYYY-MM-DD)."""
        d1 = date.fromisoformat(start)
        d2 = date.fromisoformat(end)
        return (d2 - d1).days

    @decomposition(
        intent="How many days until Christmas 2025?",
        expanded_intent="Get today's date, then calculate days between today and the target date",
    )
    def _days_until(self) -> int:
        today = self.today()
        christmas = "2025-12-25"
        return self.days_between(today, christmas)

config = LLMConfig(provider=Provider.OLLAMA, model="qwen3:4b")
agent = DateAgent(llm=config)

response = agent.run("How many days from Jan 1, 2026 to Valentine's Day 2026?")
print(f"Plan:\n{response.plan}")
print(f"Result: {response.result}")
