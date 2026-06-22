# GitHub Setup

The local repository can be created with Git immediately. Public remote creation requires either GitHub CLI authentication or a repository URL created in the browser.

## Option A - GitHub CLI

Install and authenticate `gh`, then run:

```powershell
gh auth login
gh repo create interactive-visual-variation-system --public --source . --remote origin --push
```

## Option B - Existing GitHub repo URL

Create a public empty repository on GitHub, then run:

```powershell
git remote add origin https://github.com/<user>/<repo>.git
git branch -M main
git push -u origin main
```

