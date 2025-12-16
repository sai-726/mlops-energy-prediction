# GitHub Push Guide - Step by Step

## Prerequisites
- GitHub account (create at https://github.com if you don't have one)
- Git installed on your computer

## Step 1: Create GitHub Repository

1. Go to https://github.com
2. Click the **"+"** icon (top right) → **"New repository"**
3. Fill in details:
   - **Repository name**: `mlops-energy-prediction` (or your choice)
   - **Description**: "MLOps final project - Energy consumption prediction with remote MLflow"
   - **Visibility**: **Public** (required for submission)
   - **DO NOT** check "Initialize with README" (we already have one)
4. Click **"Create repository"**
5. **Copy the repository URL** (looks like: `https://github.com/YOUR_USERNAME/mlops-energy-prediction.git`)

---

## Step 2: Install Git (If Not Installed)

Check if Git is installed:
```powershell
git --version
```

If not installed, download from: https://git-scm.com/download/win

---

## Step 3: Initialize Git Repository

Open PowerShell in your project folder:

```powershell
cd c:\Users\krish\Desktop\Kiran-MLops
```

Initialize Git:
```powershell
git init
```

Configure Git (first time only):
```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## Step 4: Review What Will Be Pushed

Check which files will be included:
```powershell
git status
```

**Important files to verify:**
- ✅ All `.py` files
- ✅ `pyproject.toml`
- ✅ `README.md`
- ✅ `.gitignore`
- ✅ Data files in `data/cleaned/`
- ❌ `.env` (should be ignored - contains secrets!)
- ❌ `.venv/` (should be ignored)

---

## Step 5: Add Files to Git

Add all files:
```powershell
git add .
```

Verify what's staged:
```powershell
git status
```

**CRITICAL CHECK:** Make sure `.env` is NOT in the list!
If you see `.env`, run:
```powershell
git reset .env
```

---

## Step 6: Commit Your Code

Create your first commit:
```powershell
git commit -m "Initial commit: Complete MLOps energy prediction project with remote MLflow"
```

---

## Step 7: Connect to GitHub

Add your GitHub repository as remote (replace with YOUR URL):
```powershell
git remote add origin https://github.com/YOUR_USERNAME/mlops-energy-prediction.git
```

Verify remote is added:
```powershell
git remote -v
```

---

## Step 8: Push to GitHub

Set the main branch and push:
```powershell
git branch -M main
git push -u origin main
```

**You'll be prompted for credentials:**
- Username: Your GitHub username
- Password: Use a **Personal Access Token** (not your password!)

### How to Create Personal Access Token:
1. Go to https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Give it a name: "MLOps Project"
4. Select scopes: Check **"repo"**
5. Click **"Generate token"**
6. **Copy the token** (you won't see it again!)
7. Use this token as your password when pushing

---

## Step 9: Verify Upload

1. Go to your GitHub repository URL in browser
2. Check that all files are visible
3. Verify `.env` is **NOT** uploaded
4. Check README.md displays correctly

---

## Step 10: Add Screenshots (After Taking Them)

After you take screenshots:

```powershell
# Create screenshots folder if not exists
mkdir screenshots

# Add screenshots
git add screenshots/
git commit -m "Add project screenshots"
git push
```

---

## Common Issues & Solutions

### Issue 1: "Permission denied"
**Solution:** Use HTTPS URL (not SSH) or set up SSH keys

### Issue 2: "Large files detected"
**Solution:** Check `.gitignore` is working:
```powershell
git rm --cached data/raw/energydata_complete.csv
git commit -m "Remove large file"
```

### Issue 3: ".env file uploaded by mistake"
**URGENT Solution:**
```powershell
git rm --cached .env
git commit -m "Remove .env file"
git push
```
Then **immediately change all credentials** in AWS and Neon!

---

## What Should Be in Your Repository

### ✅ Include:
- All `.py` source files
- `pyproject.toml`
- `README.md`
- `.gitignore`
- `.env.example`
- `data/cleaned/` (train, val, test CSVs)
- `data/drift/` (production_data.csv)
- `mlflow_setup/` (config and docs)
- `tests/`
- `screenshots/` (after taking them)
- `*.md` files (guides, documentation)

### ❌ DO NOT Include:
- `.env` (secrets!)
- `.venv/` (virtual environment)
- `__pycache__/` (Python cache)
- `*.pyc` (compiled Python)
- `mlruns/` (local MLflow runs)
- `*.log` (log files)
- Large model files (stored in S3)

---

## Final Checklist

- [ ] Repository created on GitHub
- [ ] Git initialized locally
- [ ] All files added and committed
- [ ] Remote added
- [ ] Pushed to GitHub successfully
- [ ] Verified on GitHub website
- [ ] `.env` NOT uploaded
- [ ] README.md looks good
- [ ] Repository URL copied for submission

---

## Your Repository URL

After pushing, your repository will be at:
```
https://github.com/YOUR_USERNAME/mlops-energy-prediction
```

**Copy this URL for your project submission!**

---

## Quick Commands Summary

```powershell
# One-time setup
cd c:\Users\krish\Desktop\Kiran-MLops
git init
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Add and commit
git add .
git commit -m "Initial commit: Complete MLOps project"

# Connect to GitHub
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push
git branch -M main
git push -u origin main

# Future updates
git add .
git commit -m "Update: description of changes"
git push
```

---

## Need Help?

If you encounter issues:
1. Check Git is installed: `git --version`
2. Verify remote URL: `git remote -v`
3. Check what's staged: `git status`
4. Review `.gitignore` file

Good luck! 🚀
