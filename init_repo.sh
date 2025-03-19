#!/bin/bash
# Initialize the git repository and make the first commit

# Initialize the repository
git init

# Make scripts executable
chmod +x scripts/*.py

# Add all files
git add .

# Make the initial commit
git commit -m "Initial commit: UI alignment detection with LoRA training

- Added core modules for UI alignment detection
- Implemented LoRA training pipeline
- Created inference and analysis scripts
- Added dataset preparation utilities
- Set up project structure and documentation

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

echo "Repository initialized and first commit created"