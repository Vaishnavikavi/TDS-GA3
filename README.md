# Safe AI Mailroom

FastAPI endpoint implementing the `ga5-mailroom-action-gate/v2` propose/commit envelope, canonical SHA-256 hashing, persistent SQLite evaluation/cache state, and Ed25519 receipt verification.

## Deploy to Render

1. Push this folder to a new public GitHub repository.
2. In Render, create **New + → Blueprint** and select the repository. Render reads `render.yaml`.
3. Set `OPENAI_API_KEY` in the Render service environment. Keep it secret; never put it in source code.
4. Wait for deploy, then test `https://YOUR-SERVICE.onrender.com/health`.
5. Submit the service base URL (no query, fragment, or credentials), e.g. `https://safe-ai-mailroom.onrender.com`.

## Important

- A persistent disk is configured because receipts and replay state must survive restarts. Check current Render free-tier disk/plan availability before relying on it; if persistent storage is not available on your plan, use a host with a persistent volume or a managed database.
- This is a deployable starter, not a guarantee of full marks. Hidden dossier accuracy depends on the model and its decision quality. Run the grader's Check and inspect category feedback before Save; remember Save consumes a check.
- Do not use real customer data or expose your model API key.
