# "Analyze the buttons in the provided image. Your goal is to identify any design flaws or inconsistencies that detract from a polished and professional appearance. Specifically, assess the following:

# *   **Text Alignment:** Evaluate the horizontal and vertical centering of the button text. Is the text consistently positioned within the button, or is it skewed to one side or too close to the top/bottom edges?
# *   **Button Appearance:** Assess the visual alignment of the buttons with other UI elements. Look for distortions, clipping, color inconsistencies, or any visual anomalies in the button's shape or appearance.
# *   **Overall Professionalism:** Does the button design convey a sense of polish and attention to detail, or are there visual cues that suggest misalignment, imbalance, or a lack of refinement?
# *   **Prioritize identifying subtle inconsistencies and potential areas for improvement, even if they seem minor.**
# *   **Clearly articulate the specific issue, its impact on the overall design, and a suggested solution.**"


# UI Alignment Training Dataset Requirements
# Here's a comprehensive list of screenshots you should capture for your training dataset. This collection aims to cover common UI alignment issues while keeping the dataset size manageable.
# Dataset Size Recommendation

# Minimum viable dataset: 300-500 screenshots
# Optimal dataset: 800-1200 screenshots
# Format: High-resolution PNG/JPEG images (1080p or higher)
# Total size: Approximately 1-2GB for the optimal dataset

# Screenshot Categories to Capture
# 1. Buttons (60-100 examples)

# Buttons with perfectly centered text (20-30)
# Buttons with off-center text (left/right bias) (20-30)
# Buttons with text vertically misaligned (too high/low) (20-30)
# Various button sizes and shapes in both correct and incorrect states

# 2. Form Elements (80-120 examples)

# Properly aligned input fields with labels (20-30)
# Misaligned input fields (not matching label width) (20-30)
# Form groups with consistent vs. inconsistent spacing (20-30)
# Dropdown menus with alignment issues (20-30)

# 3. Navigation Components (60-80 examples)

# Properly aligned navigation bars (15-20)
# Misaligned navigation items (15-20)
# Mobile navigation menus (correct and incorrect) (15-20)
# Tab interfaces with alignment issues (15-20)

# 4. Content Layouts (100-150 examples)

# Text blocks with proper alignment vs. text blocks with ragged edges (25-30)
# Cards/panels with proper vs. improper internal alignment (25-30)
# Grid layouts with consistent vs. inconsistent spacing (25-30)
# Lists with proper vs. improper indentation (25-30)
# Tables with aligned vs. misaligned columns (25-30)

# 5. Responsive Design Issues (80-100 examples)

# Same pages at different viewport sizes showing proper adaptation (25-30)
# Overflow issues on small screens (25-30)
# Improperly stacked elements in mobile views (25-30)
# Inconsistent margins across different devices (25-30)

# 6. Spacing Inconsistencies (60-80 examples)

# UIs with consistent padding/margins (20-25)
# UIs with inconsistent padding/margins (20-25)
# Element groups with proper vs. improper breathing room (20-25)

# 7. Color and Contrast (40-60 examples)

# Elements with proper visual hierarchy (20-30)
# Elements with poor contrast creating alignment perception issues (20-30)

# 8. Special Components (60-80 examples)

# Modal dialogues (centered vs. off-center) (15-20)
# Tooltips and popovers (properly vs. improperly positioned) (15-20)
# Notification banners (aligned vs. misaligned) (15-20)
# Loading indicators (centered vs. off-center) (15-20)

# 9. Complex Page Examples (80-100 examples)

# Dashboard pages with multiple alignment issues vs. corrected versions (30-40)
# Product pages with various alignment problems vs. corrected versions (30-40)
# Mixed-content pages with tables, forms, and various UI elements (30-40)

# Annotation Guidelines
# For each screenshot:

# Capture pairs where possible (correct version and incorrect version)
# Include brief text descriptions of what's wrong in the misaligned examples
# Consider marking problem areas with subtle highlights or arrows in a subset of the images

# This dataset structure provides a comprehensive foundation for fine-tuning your model to identify UI alignment issues across various components and contexts. The variety of examples will help the model generalize to new unseen interfaces in your application.