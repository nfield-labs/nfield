# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.0.1 (latest) | ✅ Active |
| < 0.0.1 | ❌ Not supported |

We support the latest release with security patches. When v0.1.0 is released, v0.0.1 will receive critical security fixes for 90 days before reaching end-of-life.

---

## Reporting a Vulnerability

**Do not report security vulnerabilities through public GitHub issues.** Public disclosure before a fix is available puts all users at risk.

### How to Report

Use **[GitHub Private Security Advisories](https://github.com/formatshield/formatshield/security/advisories/new)** to submit a vulnerability report. This keeps the report private until a fix is ready.

Your report should include:

1. **Description** — what the vulnerability is and what it enables
2. **Steps to reproduce** — minimal code or commands to trigger the issue
3. **Impact assessment** — who is affected and under what conditions
4. **Your suggested fix** (optional but appreciated)
5. **Your preferred credit** — how you'd like to be acknowledged

### What NOT to Report Here

The following are **out of scope** for this security policy:

- Vulnerabilities in third-party LLM provider APIs (Groq, OpenAI, Anthropic) — report to the respective providers
- Issues requiring physical access to the user's machine
- User-managed API key exposure — FormatShield never stores keys, but you are responsible for how you pass them
- Rate limit bypasses on external APIs
- Social engineering attacks

---

## Response Timeline

| Milestone | Target |
|-----------|--------|
| Acknowledgement of report | ≤ 48 hours |
| Vulnerability confirmed/denied | ≤ 5 business days |
| Patch developed (HIGH/CRITICAL) | ≤ 14 days from confirmation |
| Patch developed (MEDIUM) | ≤ 30 days from confirmation |
| Patch developed (LOW) | Next regular release |
| Public disclosure | Coordinated with reporter after patch ships |

---

## Disclosure Policy

FormatShield follows **coordinated disclosure**:

1. Reporter submits vulnerability privately
2. We confirm receipt within 48 hours
3. We investigate and develop a fix
4. We notify reporter when fix is ready
5. We release the fix
6. We publish a security advisory (for HIGH/CRITICAL)
7. Reporter may publish their own writeup 7 days after patch ships

We will never ask you to delay disclosure beyond 90 days from first report.

---

## Security Audit Tooling

FormatShield's CI runs automated security checks on every PR and weekly on `main`:

```bash
# Static analysis — run locally:
uv run bandit -r src/formatshield/ -ll

# Dependency vulnerability scan:
uv run pip-audit

# Check for secrets in code:
uv run detect-secrets scan src/
```

---

## Bug Bounty

FormatShield is an open-source project maintained by volunteers. We do not offer monetary bug bounties.

We offer:
- **Credit** in the security advisory and CHANGELOG
- **Acknowledgment** in CONTRIBUTORS.md
- Our sincere gratitude for keeping users safe

---

## Safe Harbor

We will not pursue legal action against researchers who:

- Report vulnerabilities through this policy in good faith
- Make a reasonable effort to avoid privacy violations and service disruption
- Give us reasonable time to respond before any disclosure
- Do not exploit vulnerabilities beyond what is necessary to demonstrate the issue

---

## Security Hardening Recommendations

When deploying FormatShield in production:

1. **Never log prompts** — FormatShield logs no prompts by default; keep it that way in production
2. **Use environment variables for API keys** — never hardcode keys in source code
3. **Pin dependency versions** — use `uv.lock` for reproducible, auditable builds
4. **Audit backend access** — FormatShield passes your prompts to the backend you configure; ensure it matches your data sensitivity
5. **Validate user-supplied schemas** — if schemas are user-provided, validate them before passing to FormatShield

---

*Last updated: 2026-04-13*
