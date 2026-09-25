---
name: github-only-git
description: Enforce that every Git remote operation for GeneRec uses only https://github.com/cpxjazz123/HiCurvRec.git.
---

# GitHub-only Git policy

This repository has exactly one permitted Git remote:

```text
https://github.com/cpxjazz123/HiCurvRec.git
```

## Mandatory rules

- Before any remote operation, run `git remote -v` and verify that `origin` uses the URL above.
- `clone`, `fetch`, `pull`, `push`, `ls-remote`, remote branch queries, remote tag queries, and remote configuration changes may target only this GitHub repository.
- Never use a GitLab remote or any other remote, even if it remains in local Git history or configuration.
- Work only on `main`; its upstream must be `origin/main`.
- Push completed commits with `git push origin main`. Do not push temporary or personal branches.
- Do not use `git push --force` or `git push --force-with-lease` unless the user explicitly authorizes history rewriting.
- After every push, compare the local and GitHub hashes:

  ```bash
  git rev-parse main
  git ls-remote origin refs/heads/main
  ```

  Stop if the hashes differ.

If the configured remote or upstream violates this policy, correct it before doing any other remote Git operation. Do not contact the old remote while correcting the configuration.
