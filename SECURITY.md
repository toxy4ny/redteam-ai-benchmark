# Security Policy

## Supported Versions

Security fixes target the current `main` branch. Tagged releases may be superseded by the latest source when the project moves quickly.

## Reporting a Vulnerability

Please do not open a public issue with exploit details, secrets, private model outputs, or information about real third-party targets.

Use GitHub private vulnerability reporting if it is enabled for the repository. If it is not enabled, open a minimal public issue asking for a private maintainer contact and include only:

- affected file or feature
- high-level impact
- whether you have a local reproducer

## Scope

Good reports include vulnerabilities in this repository's code, packaging, exports, configs, or documentation that could expose secrets, corrupt benchmark results, or mislead users about security-sensitive behavior.

Out of scope: vulnerabilities in benchmarked models, provider APIs, local Ollama/LM Studio/OpenWebUI deployments, OpenRouter, or third-party targets. Report those to the affected project or service.
