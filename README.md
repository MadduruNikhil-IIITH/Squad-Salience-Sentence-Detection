# project

This repository contains the project files for the CL2 project.

## Contents

- data/: dataset files (train.json, dev.json)
- menv/: Python virtual environment

## Getting started

1. Create and activate your virtual environment (if you prefer not to use the provided `menv`).
2. Install dependencies, if any, and run the project.

## GitHub repository

Follow the steps below to create a GitHub remote and push this project to GitHub:

1. Initialize a git repo:

```powershell
git init
```

2. Create a GitHub repository and link it as a remote. Using the GitHub CLI:

```powershell
gh repo create <your-username>/CL2_project --public --source . --remote origin
```

3. Push the local commits to GitHub:

```powershell
git push -u origin main
```

If you do not have gh, use the web UI and run:

```powershell
git remote add origin https://github.com/<your-username>/CL2_project.git
git branch -M main
git push -u origin main
```