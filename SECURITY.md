# Security Policy

## API keys

Webflyx RAG Studio does not store API keys in application configuration files.

On Windows, secrets are stored through Python's `keyring` integration with the
operating-system credential vault.

Do not commit `.env` files, API keys, tokens, private keys, or credentials.

## Reporting a vulnerability

Please report security vulnerabilities privately to the repository owner rather
than opening a public issue containing exploit details or credentials.

## Network behavior

Local search and embedding operations run locally.

The application connects externally only when required to:

- download the public movie dataset;
- download machine-learning model files on first use;
- make explicitly requested OpenRouter LLM requests.

Retrieved movie text is sent to OpenRouter only when the user invokes an AI
assistant feature.
