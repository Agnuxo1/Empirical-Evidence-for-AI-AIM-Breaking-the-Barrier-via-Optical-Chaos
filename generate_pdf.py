#!/usr/bin/env python3
"""
Script to generate PDF from HTML using WeasyPrint
If WeasyPrint is not available, provides instructions for manual conversion.
"""

import sys
import os

def generate_pdf():
    html_file = "Darwins_Cage_Comprehensive_Paper.html"
    pdf_file = "Darwins_Cage_Comprehensive_Paper.pdf"
    
    if not os.path.exists(html_file):
        print(f"Error: {html_file} not found!")
        return False
    
    try:
        from weasyprint import HTML, CSS
        
        print("Generating PDF from HTML...")
        HTML(filename=html_file).write_pdf(
            pdf_file,
            stylesheets=[CSS(string='@page { size: A4 portrait; margin: 2cm; }')]
        )
        
        if os.path.exists(pdf_file):
            size = os.path.getsize(pdf_file) / 1024  # KB
            print(f"✓ PDF generated successfully: {pdf_file} ({size:.1f} KB)")
            return True
        else:
            print("✗ PDF generation failed - file not created")
            return False
            
    except ImportError:
        print("=" * 70)
        print("WeasyPrint is not installed or dependencies are missing.")
        print("\nTo install WeasyPrint on Windows:")
        print("1. Install GTK+ runtime: https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer")
        print("2. Then: pip install weasyprint")
        print("\nAlternative: Use an online HTML to PDF converter or browser print:")
        print(f"  - Open {html_file} in Chrome/Edge")
        print("  - Press Ctrl+P (Print)")
        print("  - Select 'Save as PDF'")
        print("  - Set margins to 2cm and ensure 2-column layout is preserved")
        print("=" * 70)
        return False
    except Exception as e:
        print(f"Error generating PDF: {e}")
        print("\nYou can still convert the HTML manually:")
        print(f"1. Open {html_file} in a web browser")
        print("2. Use browser's Print function (Ctrl+P)")
        print("3. Select 'Save as PDF'")
        return False

if __name__ == "__main__":
    success = generate_pdf()
    sys.exit(0 if success else 1)

