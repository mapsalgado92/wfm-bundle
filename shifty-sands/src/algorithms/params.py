class Params:
    def __init__(self, type: str):
        self.type = type

    def __repr__(self) -> str:
        return f"Params(type='{self.type}')"

    def is_type(self, type: str) -> bool:
        return type == self.type


class GradientDescentParams(Params):
    def __init__(
        self,
        learning_rate: float = 0.01,
        num_iterations: int = 10000,
        snapshot_length: int = 100,
    ):
        super().__init__(type="gradient_descent")
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.snapshot_length = snapshot_length

        if 0 >= self.learning_rate:
            raise ValueError("learning_rate must be greater than 0")
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be greater than 0")


class StochasticRoundingParams(Params):
    def __init__(
        self,
        section_size: int,
        passes: int = 25,
    ):
        super().__init__(type="stochastic_rounding")
        self.section_size = section_size
        self.passes = passes
        if self.section_size <= 0:
            raise ValueError("section_size must be greater than 0")
        if self.passes <= 0:
            raise ValueError("passes must be greater than 0")
