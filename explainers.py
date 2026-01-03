import re

FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "live.com", "aol.com", "yandex.com"
}

SUSPICIOUS_PHRASES = [
    "no experience", "immediate hiring", "apply now", "quick money",
    "work from home", "refundable", "deposit", "pay to apply",
    "training materials", "western union", "send money", "earn", "earn upto",
    "must have a laptop", "visa sponsorship not required"
]

DEPOSIT_PHRASES = ["deposit", "refundable", "training fee", "training materials", "payment required", "pay to apply"]
SALARY_TERMS = ["salary", "ctc", "package", "per month", "per annum", "pa", "per year", "\$", "₹", "£"]


def _find_emails(text: str):
    return re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)


def generate_explanations(text: str, title: str = "", company: str = "", location: str = "", how_to_apply: str = "", requirements: str = "", description: str = ""):
    """Return a list of short explanation strings (human-readable) describing
    why a posting may be suspicious.

    This is intentionally rule-based (heuristic) so it is transparent and easy
    to tweak. It is not a replacement for model-based feature importance, but
    it makes common scam indicators explicit for users.
    """

    explanations = []
    lower = (text or "").lower()

    # Suspicious phrases
    matches = [p for p in SUSPICIOUS_PHRASES if p in lower]
    if matches:
        # show up to 6 short phrases
        short = ", ".join(matches[:6])
        explanations.append(f"Suspicious phrases detected: {short}")

    # Missing company information
    if not company or company.strip().lower() in {"", "n/a", "na", "unknown", "none"}:
        explanations.append("Missing company details")

    # Generic contact email
    emails = _find_emails(text)
    if emails:
        domains = [e.split("@", 1)[1].lower() for e in emails]
        if any(d in FREE_EMAIL_DOMAINS for d in domains):
            explanations.append("Uses a generic email address for contact (e.g., Gmail/Yahoo)")
    # If there's no email at all, that's a weak signal (optional)
    else:
        explanations.append("No direct contact email found")

    # Requests for payment or deposits
    if any(p in lower for p in DEPOSIT_PHRASES):
        explanations.append("Asks for payment/deposit or refundable fee — do not pay to apply")

    # Salary not provided
    if not any(s in lower for s in SALARY_TERMS):
        explanations.append("No salary or compensation details provided")

    # Extra heuristic: too short description
    if description and len(description.strip().split()) < 20:
        explanations.append("Very short description — lacks detail about the role")

    # Deduplicate maintaining order
    seen = set()
    out = []
    for e in explanations:
        if e not in seen:
            seen.add(e)
            out.append(e)

    return out
