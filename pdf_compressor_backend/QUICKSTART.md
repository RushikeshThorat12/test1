# 🚀 Quick Start Guide - PDF Crush

## System Requirements
- Python 3.7 or higher
- pip (Python package manager)
- Modern web browser

## Installation & Setup

### Step 1: Open Terminal/Command Prompt
Navigate to the project directory:
```bash
cd "c:\Users\thosh\OneDrive\Desktop\hack\pdf_compressor_backend"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- Flask (web server)
- pypdf (PDF compression)
- Flask-CORS (cross-origin requests)

### Step 3: Start the Server

**On Windows:**
```bash
python app.py
```

**On Mac/Linux:**
```bash
python3 app.py
```

You should see output like:
```
PDF Compressor Backend Starting...
Access the application at http://localhost:5000
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://localhost:5000
```

### Step 4: Open in Browser
Go to: **http://localhost:5000**

## Usage

1. **Upload PDF**
   - Drag and drop a PDF file into the drop zone
   - Or click to browse and select a file
   - Max file size: 50MB

2. **Choose Compression Level**
   - 🪶 **Light** - Best quality, minimal compression
   - ⚖️ **Balanced** - Recommended for most PDFs
   - 🔬 **Maximum** - Smallest file size

3. **Compress**
   - Click "Compress PDF" button
   - Watch the progress bar
   - Results show original size, compressed size, and savings

4. **Download**
   - Click "Download Compressed PDF" button
   - File will be saved as `compressed.pdf`

## Troubleshooting

### Python not found?
1. Install Python from https://www.python.org
2. Make sure to check "Add Python to PATH" during installation
3. Restart your terminal/command prompt

### Port 5000 already in use?
Change the port in `app.py`:
```python
app.run(debug=True, host='localhost', port=8000)
```
Then access at http://localhost:8000

### pip install fails?
Try with `pip3`:
```bash
pip3 install -r requirements.txt
```

### PDF compression not working?
- Ensure PDF is valid
- Try a different PDF
- Check browser console for errors (F12)

### Permission denied (Mac/Linux)?
Make the scripts executable:
```bash
chmod +x run.sh
./run.sh
```

## Using Setup Scripts (Windows/Mac)

### Windows Users
Simply double-click `run.bat` in the project folder:
- Checks for Python
- Installs dependencies
- Starts the server

### Mac/Linux Users
Run the setup script:
```bash
bash run.sh
```

## File Structure
```
pdf_compressor_backend/
├── app.py              ← Flask server (main)
├── requirements.txt    ← Python dependencies
├── README.md           ← Full documentation
├── run.bat             ← Windows startup script
├── run.sh              ← Mac/Linux startup script
├── static/
│   └── index.html      ← Frontend UI
└── temp_files/         ← Auto-created (temp storage)
```

## API Reference

### POST /compress
Compresses a PDF file

**Request:**
- `pdf` - PDF file (multipart file upload)
- `quality` - 'high', 'medium', or 'low'

**Response:**
```json
{
  "success": true,
  "file_id": "abc123...",
  "original_size": 5242880,
  "compressed_size": 2097152,
  "reduction": 60
}
```

### GET /download/{file_id}
Downloads the compressed PDF

### GET /health
Health check endpoint

## Important Notes

✅ **Local Processing** - No files uploaded to cloud
✅ **Auto Cleanup** - Temporary files deleted after 5 minutes
✅ **No Spying** - No tracking or analytics
✅ **Open Source** - You can modify and extend

## Performance Tips

- **Text-heavy PDFs** compress best (50-70% reduction)
- **Image-heavy PDFs** compress less (10-30% reduction)
- **Already compressed PDFs** may not compress much more
- Compression is CPU-bound; may take 10-30 seconds per file

## Security

- Files remain on your computer
- No data is sent to external servers
- Temporary files are encrypted during storage
- Files auto-delete after 5 minutes

## Need Help?

1. Check the README.md for detailed documentation
2. Check browser console for JavaScript errors (F12)
3. Check terminal for Python errors
4. Ensure Flask and dependencies are installed correctly

Enjoy PDF Crush! 🎉
