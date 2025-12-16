# Git Repository Cleanup Checklist

## Files to Remove

### 1. Unnecessary Files

```bash
git rm "New Text Document.txt"
```

## Files to Add

### 1. New Metric System

```bash
git add metrics/
git add metric_loader.py
git add config.yaml
```

### 2. Documentation

```bash
git add QUICKSTART.md
git add REFACTORING_GUIDE.md
git add example_usage.py
```

### 3. Repository Configuration

```bash
git add .gitignore
git add requirements.txt  # Updated with pyyaml
git add readme.md         # Updated with refactoring info
```

## Commit Messages

### Step 1: Remove Unnecessary Files

```bash
git commit -m "Remove unnecessary files from repository"
```

### Step 2: Add Refactored Structure

```bash
git add metrics/ metric_loader.py config.yaml
git commit -m "Add modular metric system with configuration support

- Create metrics/ directory with individual metric implementations
- Add metric_loader.py for dynamic metric loading
- Add config.yaml for centralized configuration
- Each metric is now a separate class in its own file
- All metrics inherit from AbstractMetric base class
- Maintains full backward compatibility with existing code"
```

### Step 3: Add Documentation

```bash
git add QUICKSTART.md REFACTORING_GUIDE.md example_usage.py
git commit -m "Add comprehensive documentation for refactored structure

- QUICKSTART.md: Quick start guide for new users
- REFACTORING_GUIDE.md: Detailed refactoring documentation
- example_usage.py: Working example of new metric system"
```

### Step 4: Update Repository Configuration

```bash
git add .gitignore requirements.txt readme.md
git commit -m "Update repository configuration and documentation

- Add .gitignore for Python projects
- Add pyyaml to requirements.txt
- Update readme.md with refactoring information"
```

## Verify Repository Status

```bash
# Check what's staged
git status

# Review changes
git diff --cached

# View commit history
git log --oneline -5
```

## Push to Remote

```bash
# Push to main branch
git push origin main

# Or create a new branch for the refactoring
git checkout -b refactor/modular-metrics
git push origin refactor/modular-metrics
```

## Complete Cleanup Command Sequence

```bash
# Remove unnecessary file
git rm "New Text Document.txt"

# Stage all new files
git add metrics/
git add metric_loader.py
git add config.yaml
git add QUICKSTART.md
git add REFACTORING_GUIDE.md
git add example_usage.py
git add .gitignore
git add requirements.txt
git add readme.md

# Single comprehensive commit
git commit -m "Refactor: Add modular metric system with configuration support

Major Changes:
- Create modular metric system in metrics/ directory
- Each scoring function is now a separate class
- Add config.yaml for centralized configuration
- Add metric_loader.py for dynamic metric loading
- Full backward compatibility maintained

Documentation:
- Add QUICKSTART.md for quick start guide
- Add REFACTORING_GUIDE.md for detailed documentation
- Add example_usage.py demonstrating new system

Repository Cleanup:
- Remove 'New Text Document.txt'
- Add comprehensive .gitignore
- Update requirements.txt with pyyaml
- Update readme.md with new features

All existing scripts (extract.py, evaluate_mia.py, run_pipeline.py)
continue to work without modifications."

# Push changes
git push origin main
```

## Verification Steps

After committing, verify:

1. **All metrics load correctly**:

   ```bash
   python example_usage.py
   ```

2. **Original scripts still work**:

   ```bash
   python run_pipeline.py --help
   ```

3. **Config system works**:

   ```bash
   python -c "from metric_loader import load_config; print(load_config('config.yaml')['global']['model'])"
   ```

4. **Repository is clean**:
   ```bash
   git status
   # Should show: "nothing to commit, working tree clean"
   ```

## Additional Recommendations

### Create Release Tag

```bash
git tag -a v2.0.0 -m "Version 2.0.0: Modular metric system"
git push origin v2.0.0
```

### Create Branch Protection

If using GitHub:

1. Go to Settings → Branches
2. Add rule for `main` branch
3. Require pull request reviews
4. Require status checks

### Add GitHub Actions (Optional)

Create `.github/workflows/test.yml` for automated testing:

```yaml
name: Test Metrics
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Test metric loading
        run: python example_usage.py
```

## Summary

The repository is now ready to be published with:

- ✅ Clean git history
- ✅ Modular, maintainable code structure
- ✅ Comprehensive documentation
- ✅ Backward compatibility
- ✅ Configuration-driven experiments
- ✅ Professional repository setup

Ready to share with the research community! 🚀
