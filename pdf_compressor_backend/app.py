import os
import uuid
import time
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pypdf import PdfReader, PdfWriter
from pathlib import Path
import io

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'temp_files'
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
FILE_RETENTION = 5 * 60  # 5 minutes in seconds

# Create temp folder
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# Store file metadata
files_db = {}


def cleanup_old_files():
    """Remove files older than retention time"""
    now = time.time()
    expired_ids = [
        fid for fid, data in files_db.items()
        if now - data['created'] > FILE_RETENTION
    ]
    
    for fid in expired_ids:
        try:
            file_path = files_db[fid]['path']
            if os.path.exists(file_path):
                os.remove(file_path)
            del files_db[fid]
        except Exception as e:
            print(f"Cleanup error: {e}")


def compress_pdf(pdf_bytes, quality='medium'):
    """
    Compress PDF using pypdf library
    quality: 'high' (high quality, less compression)
             'medium' (balanced)
             'low' (maximum compression)
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        writer = PdfWriter()
        
        # Append all pages from reader - this is the safe way
        writer.append_pages_from_reader(reader)
        
        # Preserve metadata if available
        if reader.metadata:
            writer.add_metadata(reader.metadata)
        
        # Write compressed PDF to bytes
        output = io.BytesIO()
        writer.write(output)
        compressed_bytes = output.getvalue()
        
        # Calculate compression stats
        original_size = len(pdf_bytes)
        compressed_size = len(compressed_bytes)
        reduction = round((1 - compressed_size / original_size) * 100) if original_size > 0 else 0
        
        return {
            'success': True,
            'data': compressed_bytes,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'reduction': reduction
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def compress_pdf_advanced(pdf_bytes, quality='medium'):
    """
    Advanced PDF compression with more aggressive optimization
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        writer = PdfWriter()
        
        # Quality levels determine compression approach
        compression_level = {
            'high': 0,      # Minimal compression
            'medium': 1,    # Medium compression
            'low': 2        # Maximum compression
        }.get(quality, 1)
        
        # Process each page
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            
            # Compress content streams
            page.compress_content_streams()
            
            # Add page to writer (now it's properly part of the writer)
            writer.add_page(page)
        
        # Preserve metadata
        if reader.metadata:
            writer.add_metadata(reader.metadata)
        
        # Write to bytes
        output = io.BytesIO()
        writer.write(output)
        compressed_bytes = output.getvalue()
        
        original_size = len(pdf_bytes)
        compressed_size = len(compressed_bytes)
        reduction = round((1 - compressed_size / original_size) * 100) if original_size > 0 else 0
        
        return {
            'success': True,
            'data': compressed_bytes,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'reduction': reduction
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"PDF compression failed: {str(e)}"
        }


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_file('static/index.html', mimetype='text/html')


@app.route('/compress', methods=['POST'])
def compress():
    """Handle PDF compression"""
    try:
        # Cleanup old files periodically
        if len(files_db) % 5 == 0:
            cleanup_old_files()
        
        # Check if file is present
        if 'pdf' not in request.files:
            return jsonify({'success': False, 'error': 'No PDF file provided'}), 400
        
        file = request.files['pdf']
        quality = request.form.get('quality', 'medium')
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'success': False, 'error': 'Only PDF files are supported'}), 400
        
        # Read file data
        pdf_data = file.read()
        
        if len(pdf_data) > MAX_FILE_SIZE:
            return jsonify({'success': False, 'error': 'File size exceeds 50MB limit'}), 400
        
        # Compress PDF
        result = compress_pdf_advanced(pdf_data, quality)
        
        if not result['success']:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Compression failed')
            }), 400
        
        # Save compressed file
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.pdf")
        
        with open(file_path, 'wb') as f:
            f.write(result['data'])
        
        # Store metadata
        files_db[file_id] = {
            'path': file_path,
            'created': time.time(),
            'original_name': file.filename
        }
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'original_size': result['original_size'],
            'compressed_size': result['compressed_size'],
            'reduction': result['reduction']
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Server error: {str(e)}"
        }), 500


@app.route('/download/<file_id>')
def download(file_id):
    """Download compressed PDF"""
    try:
        cleanup_old_files()
        
        if file_id not in files_db:
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        file_path = files_db[file_id]['path']
        
        if not os.path.exists(file_path):
            del files_db[file_id]
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        return send_file(
            file_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='compressed.pdf'
        )
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Download error: {str(e)}"
        }), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'}), 200


if __name__ == '__main__':
    print("PDF Compressor Backend Starting...")
    print("Access the application at http://localhost:5000")
    app.run(debug=True, host='localhost', port=5000)
