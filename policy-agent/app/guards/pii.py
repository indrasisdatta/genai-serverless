"""PII guard - mask personally identifiable information."""

from typing import Optional

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    _analyzer = AnalyzerEngine()
    _anonymizer = AnonymizerEngine()
    _PRESIDIO_AVAILABLE = True
except ImportError:
    _analyzer = None
    _anonymizer = None
    _PRESIDIO_AVAILABLE = False


def mask_pii(text: str) -> str:
    """
    Mask PII in text using Microsoft Presidio.

    Falls back to returning the original text if Presidio is not installed.

    Args:
        text: Input text

    Returns:
        Text with PII masked
    """
    if not _PRESIDIO_AVAILABLE or not text:
        return text

    results = _analyzer.analyze(text=text, language="en")
    anonymized = _anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text