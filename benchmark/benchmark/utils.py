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
            # Check if it its input_prompt or input_tokens
            if "input_prompt" in kwargs:
                input_tokens = TokenCounterCallback.TokenCounter.count_tokens(
                    kwargs["input_prompt"], model
                )
                input_cost = self.token_to_cost(input_tokens, model, "input")
                self.input_tokens += input_tokens
                self.input_cost += input_cost
            elif "input_tokens" in kwargs:
                input_tokens = kwargs["input_tokens"]
                input_cost = self.token_to_cost(input_tokens, model, "input")
                self.input_tokens += input_tokens
                self.input_cost += input_cost
            else:
                logging.warning("No input_tokens or input_prompt found.")

            # Check if it its output_tokens or output
            if "output_prompt" in kwargs:
                output_tokens = TokenCounterCallback.TokenCounter.count_tokens(
                    kwargs["output_prompt"], model
                )
                output_cost = self.token_to_cost(output_tokens, model, "output")
                self.output_tokens += output_tokens
                self.output_cost += output_cost

            elif "output_tokens" in kwargs:
                output_tokens = kwargs["output_tokens"]
                output_cost = self.token_to_cost(output_tokens, model, "output")
                self.output_tokens += output_tokens
                self.output_cost += output_cost
            else:
                logging.warning("No output_tokens or output found.")

            self.total_tokens = self.input_tokens + self.output_tokens
            self.total_cost = self.input_cost + self.output_cost

        except Exception as e:
            logging.error(f"Error in TokenCounterCallback: {e}")

    def __str__(self):
        """String representation of the callback."""
        return f"Tokens: {self.total_tokens} | Cost: {self.total_cost}"
