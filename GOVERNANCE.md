# FormatShield Governance

> *"In the happiness of his subjects lies the king's happiness; in their welfare his welfare."*
> — Chanakya, Arthashastra (c. 350 BCE)

This document describes how FormatShield is governed, how decisions are made, and how contributors can earn greater responsibility in the project.

---

## Mission

FormatShield's mission is to **eliminate the format tax** — the accuracy loss caused by grammar-constrained decoding — and to make this fix available to every developer through a production-quality, MIT-licensed Python library.

The project will:
- Maintain a reproducible, empirically-grounded benchmark of format tax across backends
- Implement the Think-Then-Format (TTF) algorithm reliably across all major LLM providers
- Route intelligently, measure honestly, and never claim more than the benchmarks support
- Remain permissively licensed, dependency-light, and contributor-friendly

The project will NOT:
- Build a hosted service or commercial offering under this repository
- Collect telemetry or prompt data from users
- Add dependencies that compromise the MIT license chain

---

## Roles

### User

Anyone who installs and uses FormatShield. Users drive development through bug reports, feature requests, and benchmark results.

**Rights:** File issues, participate in Discussions, share benchmark results.

### Contributor

Anyone who submits a pull request, documentation improvement, benchmark result, or bug report that is accepted.

**Rights:** All User rights, plus acknowledged in CONTRIBUTORS.md and research paper acknowledgments.

### Committer

Contributors who have had 3 or more pull requests merged. Committers have demonstrated understanding of the codebase and commitment to standards.

**Rights:** All Contributor rights, plus:
- Invited to `@formatshield/committers` GitHub team
- May review and approve pull requests
- Acknowledged in MAINTAINERS.md

**How to become a Committer:** Have 3 merged PRs. A Maintainer will invite you. If not, open a GitHub Discussion to ask.

### Maintainer

Committers who take on responsibility for the long-term health of the project.

**Rights:** All Committer rights, plus:
- Repository write access
- Can merge pull requests
- Can publish releases to PyPI
- Vote on major decisions
- Listed in MAINTAINERS.md with area of ownership

**How to become a Maintainer:** Sustained contribution over 3+ months, nominated by an existing Maintainer, approved by consensus (no objections within 7 days).

---

## Decision Making

### Minor Decisions (Lazy Consensus)

Most day-to-day decisions use **lazy consensus**: if a Maintainer proposes something and no one objects within 48 hours, it proceeds.

Examples: bug fixes, documentation improvements, new backend implementations following the protocol, dependency bumps, CI/CD improvements.

### Major Decisions (Vote)

Changes that affect the project's direction, public API, or community require a formal vote.

Examples: breaking API changes, adding/removing core dependencies, license changes, major version releases, telemetry of any kind, changes to this document.

**Vote process:**
1. Open a GitHub Discussion tagged `[RFC]`
2. 7-day comment period for community input
3. Maintainers vote: simple majority of active Maintainers (must participate within 7 days)
4. Result announced in the Discussion thread

Active Maintainer = merged a PR or participated in a discussion in the last 90 days.

---

## RFC Process

For major features or architectural changes:

1. **Draft**: Open a Discussion with `[RFC]` prefix. Include: motivation, design, drawbacks, alternatives, unresolved questions.
2. **Comment period**: Minimum 7 days.
3. **Decision**: Maintainer team votes to Accept, Reject, or request Revisions.
4. **Implementation**: Accepted RFCs become GitHub issues with `accepted-rfc` label.
5. **Merge**: Implementation must reference the RFC. Reviewed by at least one Maintainer.

---

## Conflict Resolution

1. Resolve in the issue/PR thread directly, charitably, in good faith.
2. If unresolved after 48 hours, a Maintainer mediates.
3. If still unresolved, Maintainers vote. Simple majority decides. Result is binding.
4. Code of Conduct violations: handled per CODE_OF_CONDUCT.md. Not subject to public debate.

---

## Transparency

- All technical decisions happen in public (GitHub issues, PRs, Discussions)
- Maintainer votes are recorded in the relevant Discussion thread
- Security vulnerabilities handled privately per SECURITY.md, disclosed publicly after patches ship

---

*Inspired by the governance practices of the Apache Software Foundation and the Python Software Foundation.*

*Last updated: 2026-04-13*
