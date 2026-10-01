"""Manuscript block: privacy-guarded LLM symptom annotation wrappers."""

from __future__ import annotations

import re
from typing import Any, List, Tuple

from .config import PrivacyConfig
from .privacy import require_text_allowed, sanitize_reason


SINGLE_LABEL_CATEGORIES = """
Classify the message into one category:
1. Medication Issues
2. Symptom Updates & Clinical Concerns
3. Medical Equipment, Supplies, and Home Health
4. Administrative Tasks
5. Lab Test & Imaging
6. Appointment Scheduling / Rescheduling / Cancelling
7. Caregiver Support and Logistics
8. Specialist Referral related issues
9. General Communications (confirmation, gratitude)
10. General Communications: non-medical/logistics
11. General Communications (other)
Otherwise 0.

Output format:
Classification: <0 to 11>
Reason: <reason_text>
""".strip()


def extract_response_text(response: Any) -> str:
    """Extract text from common Google GenAI response representations."""
    direct_text = getattr(response, "text", None)
    if direct_text:
        return str(direct_text)
    candidates = getattr(response, "candidates", None) or []
    parts = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def parse_single_label(content: str) -> Tuple[int, str]:
    """Parse a label in 0–11 and a sanitized rationale."""
    classification = 0
    reason = "No reason provided."
    for line in str(content).splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("classification"):
            numbers = re.findall(r"\d+", stripped)
            classification = int(numbers[0]) if numbers else 0
        elif stripped.lower().startswith("reason") and ":" in stripped:
            reason = stripped.split(":", 1)[1].strip()
    if not 0 <= classification <= 11:
        classification = 0
    return classification, sanitize_reason(reason)


def parse_multi_label(content: str, maximum_label: int = 105) -> Tuple[List[int], str]:
    """Parse up to three unique labels and a sanitized rationale."""
    classifications: List[int] = [0]
    reason = "No reason provided."
    for line in str(content).splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("classification"):
            labels: List[int] = []
            for value in map(int, re.findall(r"\d+", stripped)):
                if 0 <= value <= maximum_label and value not in labels:
                    labels.append(value)
                if len(labels) == 3:
                    break
            classifications = labels or [0]
        elif stripped.lower().startswith("reason") and ":" in stripped:
            reason = stripped.split(":", 1)[1].strip()
    return classifications, sanitize_reason(reason)


def _generate(
    client: Any,
    model_id: str,
    prompt: str,
    temperature: float,
    maximum_tokens: int,
) -> str:
    try:
        from google.genai import types
    except ImportError as error:
        raise ImportError("Install google-genai to run LLM annotation.") from error
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=maximum_tokens,
        ),
    )
    return extract_response_text(response)


def classify_single_message(
    client: Any,
    model_id: str,
    message: str,
    privacy_config: PrivacyConfig,
    temperature: float = 0.0,
    maximum_tokens: int = 2500,
) -> Tuple[int, str]:
    """Classify one message into categories 0–11."""
    require_text_allowed(privacy_config)
    system = (
        "These are patient messages sent to healthcare professionals. When a message "
        "contains medical and non-medical issues, prioritize the medical topic."
    )
    content = _generate(
        client,
        model_id,
        f"{system}\n\n{SINGLE_LABEL_CATEGORIES}\n\nMessage:\n{message}",
        temperature,
        maximum_tokens,
    )
    return parse_single_label(content)


def classify_multi_label_message(
    client: Any,
    model_id: str,
    message: str,
    category_template: str,
    privacy_config: PrivacyConfig,
    temperature: float = 0.0,
    maximum_tokens: int = 2500,
) -> Tuple[List[int], str]:
    """Assign up to three symptom labels in the range 0–105."""
    require_text_allowed(privacy_config)
    instructions = (
        "Prioritize medical topics. Use category 105 only for strictly non-medical "
        "messages. Return up to three labels only when necessary.\n"
        "Output format:\nClassification: <0 to 105>\nReason: <reason_text>"
    )
    content = _generate(
        client,
        model_id,
        f"{instructions}\n\n{category_template}\n\nMessage:\n{message}",
        temperature,
        maximum_tokens,
    )
    return parse_multi_label(content)
