# PDF Crush - PDF Compressor

A fast, modern PDF compression tool running entirely locally. Drop your PDF, choose a compression level, and download the optimized file.

## Features

- 🗜️ **Local Processing** - All compression happens on your machine, no cloud uploads
- 📊 **Real-time Stats** - See original size, compressed size, and space saved
- ⚙️ **Quality Options** - Choose between Light (best quality), Balanced, or Maximum compression
- 🎨 **Modern UI** - Beautiful dark theme with smooth animations
- ⚡ **Fast** - Optimized compression using Python's pypdf library
- 🧹 **Auto Cleanup** - Compressed files auto-delete after 5 minutes

## Quick Start

### 1. Install Python Dependencies

```bash
cd pdf_compressor_backend
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python app.py
```

The server will start at `http://localhost:5000`

### 3. Open in Browser

Go to `http://localhost:5000` in your web browser

## Project Structure

```
pdf_compressor_backend/
├── app.py                 # Flask backend server
├── requirements.txt       # Python dependencies
├── static/
│   └── index.html        # Frontend UI
└── temp_files/           # Temporary PDF storage (auto-created)
```

## API Endpoints

### `GET /`
Returns the main HTML interface

### `POST /compress`
Compresses a PDF file

**Request:**
- `pdf` (file, required) - PDF file to compress
- `quality` (string) - Compression level: 'high', 'medium', 'low'

**Response:**
```json
{
  "success": true,
  "file_id": "unique-id",
  "original_size": 5242880,
  "compressed_size": 2097152,
  "reduction": 60
}
```

### `GET /download/<file_id>`
Downloads the compressed PDF

## Configuration

Edit `app.py` to modify:
- `MAX_FILE_SIZE` - Maximum upload size (default: 50MB)
- `FILE_RETENTION` - How long to keep files before auto-deletion (default: 5 minutes)
- Port number (default: 5000)

## Compression Levels

- **Light** (high) - Minimal compression, best quality
- **Balanced** (medium) - Good balance between quality and file size
- **Maximum** (low) - Maximum compression, acceptable quality

## Requirements

- Python 3.7+
- Flask 3.0.0
- pypdf 4.0.1
- Flask-CORS 4.0.0

## Troubleshooting

### Port 5000 already in use?
Change the port in `app.py`:
```python
app.run(debug=True, host='localhost', port=8000)
```

### PDF compression not working?
- Ensure the PDF file is valid
- Check console for error messages
- Try with a smaller test PDF first

### Files not being cleaned up?
Files are automatically deleted after 5 minutes. You can adjust this in `app.py` by changing `FILE_RETENTION`.

## How Compression Works

The application uses `pypdf` library to:
1. Read and parse the PDF structure
2. Compress content streams (removes redundant data)
3. Strip unnecessary metadata
4. Preserve document integrity and readability
5. Calculate compression ratio and stats

## Performance

- **Upload**: Up to 50MB PDFs
- **Compression**: Usually < 30 seconds depending on PDF complexity
- **Download**: Instant

## Notes

- All files are processed locally - nothing is uploaded to the cloud
- Temporary files are automatically deleted after 5 minutes
- Compression effectiveness varies based on PDF content and original compression level
- Complex PDFs with images may compress less than text-only PDFs
