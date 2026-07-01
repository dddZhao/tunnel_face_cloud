from __future__ import annotations


def phase1_refinement_not_enabled() -> None:
    """Document the phase-1 boundary for stable-region constrained refinement."""
    raise NotImplementedError(
        "Stable-region constrained refinement is reserved for phase 2. "
        "Phase 1 deliberately avoids unconstrained multi-cycle ICP."
    )
