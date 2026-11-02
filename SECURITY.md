# Security policy

This repository must remain free of credentials and signing material.

- Store secrets in environment variables or a managed secret store.
- Do not commit `.env` files, OAuth client secrets, service-account JSON,
  Android keystores, passwords, model credentials, or private datasets.
- Review `git diff --cached` and run a secret scanner before every push.
- If a secret is exposed, revoke/rotate it immediately and remove it from
  Git history; deleting the file in a later commit is not sufficient.
