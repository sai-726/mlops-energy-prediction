# GitHub Push Script
# Run this in PowerShell after creating GitHub repository

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MLOps Project - GitHub Push Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Configure Git
Write-Host "[Step 1/7] Configuring Git..." -ForegroundColor Yellow
git config --global user.name "Kiran"
git config --global user.email "kiran@example.com"
Write-Host "✓ Git configured" -ForegroundColor Green
Write-Host ""

# Step 2: Initialize repository
Write-Host "[Step 2/7] Initializing Git repository..." -ForegroundColor Yellow
git init
Write-Host "✓ Repository initialized" -ForegroundColor Green
Write-Host ""

# Step 3: Check status
Write-Host "[Step 3/7] Checking files to be added..." -ForegroundColor Yellow
git status
Write-Host ""
Write-Host "⚠️  IMPORTANT: Check the list above!" -ForegroundColor Red
Write-Host "⚠️  Make sure .env is NOT in the list!" -ForegroundColor Red
Write-Host ""
$continue = Read-Host "Does the list look good? (y/n)"
if ($continue -ne "y") {
    Write-Host "Stopping. Please review the files." -ForegroundColor Red
    exit
}

# Step 4: Add files
Write-Host "[Step 4/7] Adding files..." -ForegroundColor Yellow
git add .
Write-Host "✓ Files added" -ForegroundColor Green
Write-Host ""

# Step 5: Commit
Write-Host "[Step 5/7] Committing code..." -ForegroundColor Yellow
git commit -m "Initial commit: Complete MLOps energy prediction project with remote MLflow"
Write-Host "✓ Code committed" -ForegroundColor Green
Write-Host ""

# Step 6: Add remote
Write-Host "[Step 6/7] Adding GitHub remote..." -ForegroundColor Yellow
Write-Host ""
Write-Host "⚠️  BEFORE CONTINUING:" -ForegroundColor Red
Write-Host "1. Go to https://github.com" -ForegroundColor Yellow
Write-Host "2. Create a new repository" -ForegroundColor Yellow
Write-Host "3. Name it: mlops-energy-prediction" -ForegroundColor Yellow
Write-Host "4. Make it PUBLIC" -ForegroundColor Yellow
Write-Host "5. DO NOT initialize with README" -ForegroundColor Yellow
Write-Host "6. Copy the repository URL" -ForegroundColor Yellow
Write-Host ""
$repoUrl = Read-Host "Paste your GitHub repository URL here"

git remote add origin $repoUrl
Write-Host "✓ Remote added" -ForegroundColor Green
Write-Host ""

# Step 7: Push
Write-Host "[Step 7/7] Pushing to GitHub..." -ForegroundColor Yellow
Write-Host ""
Write-Host "⚠️  You will be asked for credentials:" -ForegroundColor Red
Write-Host "   Username: Your GitHub username" -ForegroundColor Yellow
Write-Host "   Password: Use Personal Access Token (NOT your password!)" -ForegroundColor Yellow
Write-Host "   Create token at: https://github.com/settings/tokens" -ForegroundColor Yellow
Write-Host ""
$push = Read-Host "Ready to push? (y/n)"
if ($push -eq "y") {
    git branch -M main
    git push -u origin main
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "✓ SUCCESS! Code pushed to GitHub!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
} else {
    Write-Host "Push cancelled. Run 'git push -u origin main' when ready." -ForegroundColor Yellow
}
