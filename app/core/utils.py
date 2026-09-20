# Lines injected by the inference gateway into model completions — never
# let them reach users or reports.
_GATEWAY_NOISE_MARKERS = ("Karpathy", "multica-ai", "karpathy-skills")


def strip_gateway_noise(text: str) -> str:
    """Drop gateway-injected notice lines (and 💡 notices) from model output."""
    lines = [
        line
        for line in (text or "").splitlines()
        if not any(marker in line for marker in _GATEWAY_NOISE_MARKERS)
        and not line.lstrip().startswith("💡")
    ]
    return "\n".join(lines).strip()
