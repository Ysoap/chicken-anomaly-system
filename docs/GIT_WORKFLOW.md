# ?? Git Workflow & Collaboration Guidelines

## Branch Strategy

```
main (production-ready)
  ?
develop (integration branch)
  ? ? ? ?
  ¦ ¦ ¦ +-- feature/mobile-app (Pandu)
  ¦ ¦ +---- feature/stage3-anomaly (Ysoap)
  ¦ +------ feature/stage2-behavior (Malik)
  +-------- feature/stage1-detection (Azis)
```

### Branch Naming Convention

- main: Production-ready code only (merge from develop after weekly integration)
- develop: Integration branch (merge feature branches here on Friday)
- eature/<stage>-<description>: Individual feature branches

**Examples:**
- eature/stage1-detection (Azis)
- eature/stage2-behavior (Malik)
- eature/stage3-anomaly (Ysoap)
- eature/mobile-app (Pandu)
- eature/firebase-integration (Ysoap)
- hotfix/stage1-fps-bug (critical fixes)

---

## Daily Workflow

### Morning Standup (9:00 AM, 15 minutes)

**Format:**
1. What I did yesterday
2. What I'll do today
3. Blockers (if any)

**Post standup updates in Discord:**
```
[Azis] ? Yesterday: Trained RT-DETR on ChickenDet
[Azis] ?? Today: Integrate ByteTrack, test on sample video
[Azis] ? Blockers: None
```

### Work on Feature Branch

```bash
# Start of day: sync with remote
git checkout feature/stage1-detection
git pull origin feature/stage1-detection

# Make changes
# ... code ...

# Commit frequently (atomic commits)
git add backend/stage1_detection/train.py
git commit -m "[Stage1] Add RT-DETR training script with ChickenDet loader"

# Push to remote daily (backup)
git push origin feature/stage1-detection
```

---

## Commit Message Convention

**Format:**
```
[Component] Short description (max 70 chars)

Optional detailed explanation if needed.
```

**Component Tags:**
- [Stage1] - Detection & tracking (Azis)
- [Stage2] - Behavior classification (Malik)
- [Stage3] - Anomaly detection (Ysoap)
- [Mobile] - Android app (Pandu)
- [API] - FastAPI backend (Ysoap)
- [Docs] - Documentation
- [Tests] - Unit tests
- [Infra] - Infrastructure (MediaMTX, Firebase)

**Examples:**
```
[Stage1] Add ByteTrack integration with Kalman filter
[Stage2] Implement 15->4 class mapping for ChickenAct
[Stage3] Extract 12 custom features from tracking data
[Mobile] Add Firebase push notification handler
[API] Implement /api/v1/anomaly/current endpoint
[Docs] Update ANALOGI_PIPELINE with VideoMAE explanation
[Tests] Add unit tests for feature extractor
[Infra] Configure MediaMTX RTSP server
```

**Bad Examples (avoid):**
```
Update code
Fixed bug
WIP
asdf
```

---

## Weekly Integration (Friday Evening)

### Step 1: Create Pull Request (Individual)

```bash
# Before PR: ensure code is clean and tested
git checkout feature/stage1-detection
git pull origin develop  # Sync with latest develop

# If conflicts, resolve them
git merge develop
# ... resolve conflicts ...
git commit -m "[Stage1] Merge develop and resolve conflicts"

# Push to remote
git push origin feature/stage1-detection
```

**Create PR on GitHub:**
1. Go to https://github.com/Ysoap/chicken-anomaly-system/pulls
2. Click "New Pull Request"
3. Base: develop, Compare: eature/stage1-detection
4. Fill PR template (see below)
5. Assign reviewers: Ysoap (Tech Lead)

### Step 2: PR Template

```markdown
## ?? Changes
Brief description of what you implemented this week.

## ?? Testing
- [ ] Manual testing performed (describe scenarios)
- [ ] Unit tests added (list files)
- [ ] No new warnings or errors

## ?? Progress
- [x] Task A completed
- [x] Task B completed
- [ ] Task C in progress (will finish next week)

## ?? Blockers
- None / Describe any blockers

## ?? Screenshots (Mobile PRs only)
(Attach screenshots if applicable)

## ?? Notes
Any additional context or notes for reviewers.
```

### Step 3: Code Review (Ysoap)

**Review Checklist:**
- [ ] Code follows project structure
- [ ] No hardcoded secrets or credentials
- [ ] Comments/docstrings for complex logic
- [ ] No merge conflicts with develop
- [ ] Tests pass (if applicable)

**Review Comments:**
- Use GitHub review features (Comment, Approve, Request Changes)
- Be constructive and specific

### Step 4: Merge to Develop (Friday Evening)

```bash
# Ysoap merges approved PRs
git checkout develop
git merge --no-ff feature/stage1-detection
git push origin develop
```

### Step 5: Integration Testing (Weekend)

**Saturday-Sunday:**
- Test integration on develop branch
- Run full pipeline if possible
- Document any integration issues in GitHub Issues

### Step 6: Merge to Main (Sunday Night)

**If develop is stable:**
```bash
git checkout main
git merge --no-ff develop
git push origin main
git tag -a v0.1.0 -m "Week 1 milestone"
git push origin v0.1.0
```

---

## Handling Conflicts

### Scenario: Merge Conflict on PR

```bash
git checkout feature/stage1-detection
git pull origin develop  # Get latest develop

# Conflicts appear
# Edit conflicted files manually
# Look for <<<<<<< HEAD markers

git add <resolved_files>
git commit -m "[Stage1] Resolve merge conflicts with develop"
git push origin feature/stage1-detection
```

**Tips:**
- Communicate with team if conflict is complex
- Don't overwrite others' work without discussion
- Test after resolving conflicts

---

## Hotfix Workflow (Critical Bugs)

If production (main) has critical bug:

```bash
# Create hotfix branch from main
git checkout main
git checkout -b hotfix/stage1-fps-bug

# Fix bug
# ... code ...

git commit -m "[Hotfix] Fix Stage 1 FPS bottleneck"
git push origin hotfix/stage1-fps-bug

# Create PR to main (skip develop)
# After merge, also merge main -> develop
```

---

## Git Best Practices

### ? DO:
- Commit frequently (atomic commits)
- Write descriptive commit messages
- Push to remote daily
- Pull before starting work
- Test before committing
- Use branches for all features

### ? DON'T:
- Commit directly to main
- Force push (git push -f) unless absolutely necessary
- Commit secrets, API keys, or credentials
- Use git add . blindly (review changes first)
- Leave commented-out code
- Commit large binary files (use Git LFS)

---

## Git Commands Cheat Sheet

```bash
# Clone repository
git clone https://github.com/Ysoap/chicken-anomaly-system.git

# Create feature branch
git checkout -b feature/my-feature

# Check status
git status

# Stage specific files
git add backend/api/main.py

# Commit
git commit -m "[API] Add health check endpoint"

# Push to remote
git push origin feature/my-feature

# Pull latest changes
git pull origin develop

# View commit history
git log --oneline -10

# View diff
git diff

# Discard local changes (careful!)
git checkout -- <file>

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Stash changes temporarily
git stash
git stash pop
```

---

## Communication

**GitHub Issues:**
- Use for bugs, feature requests, discussions
- Tag with labels: ug, enhancement, question, documentation

**Discord/Slack:**
- Daily standup updates
- Quick questions
- Integration coordination

**Weekly Sync (Friday 5 PM):**
- Review PRs together
- Discuss next week's plan
- Celebrate wins ??

---

**Questions?** Ask Ysoap or create a GitHub Issue with question label.