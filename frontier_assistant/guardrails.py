import re
from typing import Tuple

# Patterns that should never be processed
_BLOCKLIST = re.compile(
    r"\b("
    r"ignore (all |previous |prior |your |the )?(previous |prior |all |your )?instructions?"
    r"|jailbreak"
    r"|DAN mode"
    r"|act as (if you have no|an? (evil|uncensored|unrestricted))"
    r"|pretend (you (are|have) no|there (are|is) no) (rules?|restrictions?|guidelines?)"
    r"|bypass (your |all )?(safety|filter|content|restrictions?)"
    r"|override (your |all )?(safety|filter|content|restrictions?)"
    r"|write (a |the )?(bomb|weapon|malware|ransomware|exploit|virus)"
    r"|how to (make|build|create|synthesize) (a )?(bomb|drug|weapon|poison|malware)"
    r"|child (sexual|porn|exploit)"
    r"|CSAM"
    r")\b",
    re.IGNORECASE,
)


def check_input(text: str) -> Tuple[bool, str]:
    """Returns (is_safe, reason). If not safe, reason explains why."""
    if _BLOCKLIST.search(text):
        return False, "Input blocked: matched harmful content pattern."
    if len(text) > 4000:
        return False, "Input too long (max 4000 characters)."
    return True, ""


def check_output(text: str) -> Tuple[bool, str]:
    """Basic output sanity check."""
    if not text or not text.strip():
        return False, "Empty response from model."
    return True, ""
