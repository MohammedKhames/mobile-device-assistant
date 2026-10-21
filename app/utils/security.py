"""Checks untrusted device text before it reaches the language model."""

SUSPICIOUS_PHRASES = ("ignore previous", "system prompt", "you are now")


class PromptInjectionError(ValueError):
    """Raised when device text contains a suspicious instruction phrase."""


def check_prompt_injection(text: str) -> None:
    """Reject the simple injection phrases required by the assignment."""
    normalized_text = " ".join(text.casefold().split())

    if any(phrase in normalized_text for phrase in SUSPICIOUS_PHRASES):
        raise PromptInjectionError(
            "Input contains a suspicious instruction phrase."
        )