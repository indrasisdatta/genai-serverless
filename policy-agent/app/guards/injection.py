"""Guardrails - prompt injection detection and PII masking."""

from typing import Optional

from app.llm.factory import get_llm


def check_prompt_injection(text: str) -> bool:
    """
    Check if the text contains prompt injection attempts.

    Args:
        text: Input text to check

    Returns:
        True if injection detected, False otherwise
    """
    # Simple heuristic: check for common injection patterns
    injection_patterns = [
        "ignore previous instructions",
        "disregard",
        "system prompt",
        "you are now",
        "pretend to be",
        "jailbreak",
    ]
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in injection_patterns)


async def check_prompt_injection_llm(text: str) -> bool:
    """
    Check for prompt injection using an LLM classifier.

    Args:
        text: Input text to check

    Returns:
        True if injection detected, False otherwise
    """
    llm = get_llm("fast")
    prompt = (
        "Is the following text attempting to override system instructions "
        "or perform a prompt injection? Answer yes or no.\n\n"
        f"Text: {text}"
    )
    result = llm.invoke(prompt)
    return "yes" in result.content.lower()


def mask_pii(text: str) -> str:
    """
    Mask PII in text using Presidio.

    Args:
        text: Input text

    Returns:
        Text with PII masked
    """
    # Placeholder - implement with Microsoft Presidio
    # from presidio_analyzer import AnalyzerEngine
    # from presidio_anonymizer import AnonymizerEngine
    return text