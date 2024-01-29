import logging
import tiktoken
import anthropic

PRICE_NUM_TOKENS = 1000

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


def encoding_for_model(model: str):
    return tiktoken.encoding_for_model(model)

def count_tokens(text: str, model: str) -> int:
    if "claude" in model:
        return anthropic.Anthropic().count_tokens(text)

    enc = encoding_for_model(model)
    return len(enc.encode(text))


class TokenCounterCallback:
    """Callback to count the number of tokens used in a generation."""

    TOKEN_PRICES = {
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "claude-2": {"input": 0.008, "output": 0.024},
    }

    def __init__(self) -> None:
        """Initialize the callback."""
        self.cost_dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_cost": 0,
            "output_cost": 0,
            "total_cost": 0
        }

    @staticmethod
    def token_to_cost(tokens: int, model: str, tokens_type: str) -> float:
        """Converts a number of tokens to a cost in dollars."""
        return tokens / PRICE_NUM_TOKENS * TokenCounterCallback.TOKEN_PRICES[model][tokens_type]

    def calculate_cost(self, tokens_type: str, model: str, **kwargs) -> None: 
        # Check if it its prompt or tokens are passed in
        prompt_key = f"{tokens_type}_prompt"
        token_key = f"{tokens_type}_tokens"
        if prompt_key in kwargs:
            tokens = count_tokens(kwargs[prompt_key], model)
        elif token_key in kwargs:
            tokens = kwargs[token_key]
        else:
            logging.warning(f"No {token_key}_tokens or {tokens_type}_prompt found.")
        cost = self.token_to_cost(tokens, model, tokens_type)
        self.cost_dict[token_key] += tokens
        self.cost_dict[f"{tokens_type}_cost"] += cost

    def __call__(self, model: str, **kwargs) -> None:
        """Callback to count the number of tokens used in a generation."""
        if model not in list(TokenCounterCallback.TOKEN_PRICES.keys()):
            raise ValueError(f"Model {model} not supported.")
        try:
            self.calculate_cost("input", model, **kwargs)
            self.calculate_cost("output", model, **kwargs)
            self.cost_dict["total_tokens"] = self.cost_dict["input_tokens"] + self.cost_dict["output_tokens"]
            self.cost_dict["total_cost"] = self.cost_dict["input_cost"] + self.cost_dict["output_cost"]
        except Exception as e:
            logging.error(f"Error in TokenCounterCallback: {e}")

    def __str__(self):
        """String representation of the callback."""
        return f"Tokens: {self.total_tokens} | Cost: {self.total_cost}"
