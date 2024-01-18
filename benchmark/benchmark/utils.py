import logging
import tiktoken
import anthropic


def get_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class TokenCounter:
    """
    A class to handle token encoding and counting for different GPT models.
    """

    @staticmethod
    def encoding_for_model(model: str):
        return tiktoken.encoding_for_model(model)

    @staticmethod
    def count_tokens(text: str, model: str) -> int:
        if "claude" in model:
            return anthropic.Anthropic().count_tokens(text)

        enc = TokenCounter.encoding_for_model(model)
        return len(enc.encode(text))


class TokenCounterCallback:
    """Callback to count the number of tokens used in a generation."""

    TOKEN_PRICES = {
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "claude-2": {"input": 0.008, "output": 0.024},
    }
    TokenCounter = TokenCounter()

    def __init__(self) -> None:
        """Initialize the callback."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.input_cost = 0
        self.output_cost = 0
        self.total_cost = 0

    @staticmethod
    def token_to_cost(tokens: int, model: str, tokens_type: str) -> float:
        """Converts a number of tokens to a cost in dollars."""
        return tokens / 1000 * TokenCounterCallback.TOKEN_PRICES[model][tokens_type]

    def __call__(self, model: str, **kwargs) -> None:
        """Callback to count the number of tokens used in a generation."""
        if model not in list(TokenCounterCallback.TOKEN_PRICES.keys()):
            raise ValueError(f"Model {model} not supported.")
        try:
            if "claude" in model:
                input_prompt = kwargs["input_prompt"]
                output_prompt = kwargs["output_prompt"]
                self.input_tokens = self.TokenCounter.count_tokens(input_prompt, model)
                self.output_tokens = self.TokenCounter.count_tokens(
                    output_prompt, model
                )
                self.total_tokens = self.input_tokens + self.output_tokens
                self.input_cost = self.token_to_cost(self.input_tokens, model, "input")
                self.output_cost = self.token_to_cost(
                    self.output_tokens, model, "output"
                )
                self.total_cost = self.input_cost + self.output_cost

            else:
                self.input_tokens += kwargs["input_tokens"]
                self.output_tokens += kwargs["output_tokens"]
                self.total_tokens += kwargs["total_tokens"]
                self.input_cost = self.token_to_cost(self.input_tokens, model, "input")
                self.output_cost = self.token_to_cost(
                    self.output_tokens, model, "output"
                )
                self.total_cost = self.input_cost + self.output_cost

        except Exception as e:
            logging.error(f"Error in TokenCounterCallback: {e}")

    def __str__(self):
        """String representation of the callback."""
        return f"Tokens: {self.total_tokens} | Cost: {self.total_cost}"
