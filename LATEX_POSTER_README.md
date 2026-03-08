# LaTeX Poster Compilation Guide

## 📄 Files Created

1. **poster_latex.tex** - Full-featured version with beamerposter theme
2. **poster_simple.tex** - Simple version with basic packages (RECOMMENDED)

---

## 🚀 Quick Start - Compile the Poster

### **Option 1: Online (Easiest - No Installation)**

**Overleaf** (Recommended):
1. Go to https://www.overleaf.com
2. Create free account
3. Click "New Project" → "Upload Project"
4. Upload `poster_simple.tex` (or `poster_latex.tex`)
5. Click "Recompile" button
6. Download PDF

**Advantages**: 
- No installation needed
- All packages available
- Automatic compilation
- Free for basic use

---

### **Option 2: Local Compilation (Windows)**

#### **Step 1: Install LaTeX**
Download and install **MiKTeX** or **TeX Live**:
- **MiKTeX**: https://miktex.org/download (Recommended for Windows)
- **TeX Live**: https://www.tug.org/texlive/ (Full installation)

#### **Step 2: Install Required Packages** (MiKTeX auto-installs, but verify):
```
a0poster
geometry
multicol
graphicx
booktabs
xcolor
tikz
tcolorbox
qrcode
hyperref
```

#### **Step 3: Compile**

**Using Command Line**:
```cmd
cd "c:\Users\GUDA AVINASH REDDY\Downloads\batteries\untitled folder"
pdflatex poster_simple.tex
pdflatex poster_simple.tex
```
(Run twice to resolve references)

**Using TeXworks** (comes with MiKTeX):
1. Open `poster_simple.tex` in TeXworks
2. Select "pdfLaTeX" from dropdown
3. Click green "Typeset" button
4. PDF will be generated

**Using VS Code**:
1. Install extension: "LaTeX Workshop"
2. Open `poster_simple.tex`
3. Press Ctrl+Alt+B (Build)
4. PDF opens automatically

---

## 📐 Poster Specifications

- **Size**: A0 Portrait (841mm × 1189mm / 33.1" × 46.8")
- **Orientation**: Portrait (vertical)
- **Layout**: 3 columns
- **Colors**: 
  - Primary Blue: #2E5090
  - Accent Orange: #FF6B35
  - Success Green: #06D6A0
- **Fonts**: 
  - Title: Very Large
  - Headers: Large
  - Body: Normal (scaled for A0)

---

## ✏️ Customization

### **Add Your Information**

Edit these lines in the .tex file:

```latex
% Line 65 (poster_simple.tex) - Author name
{\LARGE [Your Name] \quad [Co-authors]}

% Line 66 - Institution
{\Large [Department] $\cdot$ [University/Institution]}

% Line 67 - Contact
{\large \texttt{[email@university.edu]} $\cdot$ \texttt{github.com/...}}
```

### **Add Screenshots**

To insert your dashboard screenshots:

```latex
% Add after line 100 (in any box)
\begin{center}
\includegraphics[width=0.9\textwidth]{transformer_prediction.png}
\end{center}
```

Make sure your images are in the same folder as the .tex file.

### **Change Colors**

```latex
% Lines 21-24 - Edit RGB values
\definecolor{primaryblue}{RGB}{46,80,144}      % Change to your color
\definecolor{accentorange}{RGB}{249,115,34}    % Change to your color
```

---

## 📊 Adding Charts/Images

### **Step 1: Save your screenshots**
Place these files in the same folder as poster_simple.tex:
- `transformer_prediction.png`
- `model_comparison.png`
- `compare_actual.png`
- `ensemble_prediction.png`

### **Step 2: Insert in LaTeX**

Add this code in any section:

```latex
\begin{center}
\includegraphics[width=0.85\textwidth]{model_comparison.png}
\\[0.3cm]
\textit{Figure: Side-by-side model comparison}
\end{center}
```

### **Recommended Positions**:
- Column 1: System architecture diagram (already has TikZ version)
- Column 2: Add screenshot of Transformer predictions
- Column 3: Add Model Comparison screenshot
- Bottom: Add all 4 tabs as thumbnail gallery

---

## 🎨 Advanced Customization

### **Add More Boxes**

```latex
\begin{bluebox}[title=Your Title Here]
\large
Your content here
\begin{itemize}
    \item Point 1
    \item Point 2
\end{itemize}
\end{bluebox}
```

### **Add Tables**

```latex
\begin{table}
\centering
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Value 1} & \textbf{Value 2} \\
\midrule
RMSE & 0.025V & 0.031V \\
MAE & 0.018V & 0.023V \\
\bottomrule
\end{tabular}
\end{table}
```

### **Change Layout to 2 or 4 Columns**

```latex
% For 2 columns:
\begin{multicols}{2}
% content
\end{multicols}

% For 4 columns:
\begin{multicols}{4}
% content
\end{multicols}
```

---

## 🐛 Troubleshooting

### **Error: "File not found: a0poster.cls"**
**Solution**: Install the `a0poster` package
- MiKTeX: Will auto-prompt to install
- Manual: Download from CTAN and place in TeX directory

### **Error: "qrcode package not found"**
**Solution**: Either:
1. Install `qrcode` package, OR
2. Comment out QR code lines:
```latex
% \usepackage{qrcode}
% ...
% \qrcode[height=1.5cm]{...}
```
And replace with text:
```latex
GitHub: github.com/Avinash143-ay
```

### **Error: "Dimension too large"**
**Solution**: A0 is very large. If local compilation has memory issues:
- Use Overleaf (handles large documents better)
- Or reduce to A1 size by changing:
```latex
\documentclass[final,a1paper,portrait]{a0poster}
```

### **Images not showing**
**Solution**:
1. Ensure images are in same folder as .tex file
2. Use relative paths: `./images/file.png`
3. Check file extensions (PNG, JPG, PDF supported)

### **Text too small/large**
**Solution**: Adjust scaling:
```latex
% In poster_latex.tex, line 8:
\usepackage[scale=1.24]{beamerposter}  % Increase to 1.4 for larger fonts

% In poster_simple.tex, add after \begin{document}:
\large  % or \Large or \LARGE
```

---

## 📤 Exporting for Printing

### **Step 1: Generate PDF**
Compile the LaTeX file → Output is `poster_simple.pdf`

### **Step 2: Check PDF Properties**
- Right-click PDF → Properties
- Page Size should show: 841 × 1189 mm (A0)

### **Step 3: Send to Print Shop**
**Provide print shop with**:
- `poster_simple.pdf`
- **Size**: A0 (841mm × 1189mm)
- **Orientation**: Portrait
- **Material**: Matte or glossy paper (your choice)
- **Color**: Full color (CMYK)
- **Resolution**: 150-300 DPI (PDF is vector, so quality is good)

**Typical cost**: ₹500-800 for A0 color poster in India

### **Alternative: Digital Display**
If presenting on laptop:
- Open `poster_simple.pdf` in full screen (F11 in Adobe)
- Or export as high-res PNG for PowerPoint embedding

---

## 🎯 Differences Between Two Versions

### **poster_latex.tex** (Advanced)
✅ Uses beamerposter theme for conference-style layout  
✅ More professional columnar layout  
✅ Better typography control  
⚠️ Requires more packages  
⚠️ May have compatibility issues  

### **poster_simple.tex** (Recommended)
✅ Basic packages only (higher compatibility)  
✅ Works on most systems  
✅ Easier to customize  
✅ Recommended for beginners  
⚠️ Slightly less polished look  

**Recommendation**: Start with `poster_simple.tex`. If you want more polish and have LaTeX experience, try `poster_latex.tex`.

---

## 📝 Quick Checklist

Before compiling:
- [ ] Update author name (line 65)
- [ ] Update institution (line 66)
- [ ] Update email (line 67)
- [ ] Add your screenshots (optional)
- [ ] Test compile (pdflatex)
- [ ] Fix any errors
- [ ] Compile again (for references)
- [ ] Check PDF output
- [ ] Send to print or present digitally

---

## 🆘 Still Having Issues?

### **Option 1: Use Overleaf** (Easiest)
Upload the .tex file to Overleaf - it handles everything automatically.

### **Option 2: Use PowerPoint Instead**
If LaTeX is too complex:
1. Use the content from `POSTER_A0_BATTERY_DIGITAL_TWIN.md`
2. Create poster in PowerPoint (simpler)
3. Set size to A0: Design → Slide Size → Custom (84.1 × 118.9 cm)

### **Option 3: Ask for Help**
Contact someone with LaTeX experience at your institution:
- Library (often has LaTeX support)
- CS/Engineering department
- Fellow students who've made posters

---

## 🎓 Learning Resources

- **LaTeX Basics**: https://www.overleaf.com/learn
- **Beamer Posters**: https://www.overleaf.com/gallery/tagged/poster
- **TikZ Diagrams**: https://www.overleaf.com/learn/latex/TikZ_package
- **Tables**: https://www.tablesgenerator.com/

---

## ✅ Final Output

After successful compilation, you'll have:
- **poster_simple.pdf** (or poster_latex.pdf)
- Size: A0 (841mm × 1189mm)
- Ready to print or display digitally
- Professional research poster format

**Good luck with your presentation on Monday! 🚀**

---

**Last Updated**: March 7, 2026  
**Files**: poster_latex.tex, poster_simple.tex  
**Recommended**: Use poster_simple.tex with Overleaf for easiest results
