from opensymbolicai.blueprints import PlanExecute
from opensymbolicai.core import decomposition, primitive
from opensymbolicai.llm import LLMConfig, Provider


class UnitConverter(PlanExecute):
    """An agent that converts between cooking/volume measurements."""

    # Teaspoons <-> Tablespoons (3 tsp = 1 tbsp)
    @primitive(read_only=True)
    def tsp_to_tbsp(self, tsp: float) -> float:
        """Convert teaspoons to tablespoons."""
        return tsp / 3

    @primitive(read_only=True)
    def tbsp_to_tsp(self, tbsp: float) -> float:
        """Convert tablespoons to teaspoons."""
        return tbsp * 3

    # Tablespoons <-> Cups (16 tbsp = 1 cup)
    @primitive(read_only=True)
    def tbsp_to_cups(self, tbsp: float) -> float:
        """Convert tablespoons to cups."""
        return tbsp / 16

    @primitive(read_only=True)
    def cups_to_tbsp(self, cups: float) -> float:
        """Convert cups to tablespoons."""
        return cups * 16

    # Cups <-> Pints (2 cups = 1 pint)
    @primitive(read_only=True)
    def cups_to_pints(self, cups: float) -> float:
        """Convert cups to pints."""
        return cups / 2

    @primitive(read_only=True)
    def pints_to_cups(self, pints: float) -> float:
        """Convert pints to cups."""
        return pints * 2

    # Pints <-> Quarts (2 pints = 1 quart)
    @primitive(read_only=True)
    def pints_to_quarts(self, pints: float) -> float:
        """Convert pints to quarts."""
        return pints / 2

    @primitive(read_only=True)
    def quarts_to_pints(self, quarts: float) -> float:
        """Convert quarts to pints."""
        return quarts * 2

    # Quarts <-> Gallons (4 quarts = 1 gallon)
    @primitive(read_only=True)
    def quarts_to_gallons(self, quarts: float) -> float:
        """Convert quarts to gallons."""
        return quarts / 4

    @primitive(read_only=True)
    def gallons_to_quarts(self, gallons: float) -> float:
        """Convert gallons to quarts."""
        return gallons * 4

    # Cups <-> Milliliters (1 cup = 236.588 ml)
    @primitive(read_only=True)
    def cups_to_ml(self, cups: float) -> float:
        """Convert cups to milliliters."""
        return cups * 236.588

    @primitive(read_only=True)
    def ml_to_cups(self, ml: float) -> float:
        """Convert milliliters to cups."""
        return ml / 236.588

    # Milliliters <-> Liters (1000 ml = 1 L)
    @primitive(read_only=True)
    def ml_to_liters(self, ml: float) -> float:
        """Convert milliliters to liters."""
        return ml / 1000

    @primitive(read_only=True)
    def liters_to_ml(self, liters: float) -> float:
        """Convert liters to milliliters."""
        return liters * 1000
    
    # # Gallons <-> Hogsheads (63 gallons = 1 hogshead)
    # @primitive(read_only=True)
    # def gallons_to_hogsheads(self, gallons: float) -> float:
    #     """Convert gallons to hogsheads."""
    #     return gallons / 63

    # @primitive(read_only=True)
    # def hogsheads_to_gallons(self, hogsheads: float) -> float:
    #     """Convert hogsheads to gallons."""
    #     return hogsheads * 63

    @decomposition(
        intent="Convert 2 gallons to cups",
        expanded_intent="First convert gallons to quarts, then quarts to pints, then pints to cups",
    )
    def _example_multi_step(self) -> float:
        quarts = self.gallons_to_quarts(2)
        pints = self.quarts_to_pints(quarts)
        return self.pints_to_cups(pints)

    # @decomposition(
    #     intent="Convert 4 tablespoons of honey to milliliters and 2 quarts of juice to cups",
    #     expanded_intent="Convert 4 tablespoons to cups then to milliliters for honey; convert 2 quarts to pints then to cups for juice. Return both results in a dictionary with labels and units.",
    # )
    # def _dual_conversion(self) -> dict:
    #     # Convert 4 tablespoons of honey to milliliters
    #     cups_from_tbsp = self.tbsp_to_cups(4)
    #     honey_ml = self.cups_to_ml(cups_from_tbsp)

    #     # Convert 2 quarts of juice to cups
    #     pints_from_quarts = self.quarts_to_pints(2)
    #     juice_cups = self.pints_to_cups(pints_from_quarts)
    #     result = {
    #         "honey": {"value": honey_ml, "unit": "milliliters"},
    #         "juice": {"value": juice_cups, "unit": "cups"},
    #     }
    #     return result


if __name__ == "__main__":
    config = LLMConfig(provider=Provider.OLLAMA, model="gpt-oss:20b")
    agent = UnitConverter(llm=config)

    response = agent.run("20 gallons in liters")

    # response = agent.run("Convert 3 cups of milk to liters and separately convert 2 beer pints to teaspoons")
    
    print(f"Plan:\n{response.plan}")
    print(f"Result: {response.result}")
