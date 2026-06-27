import express from 'express';
import multer from 'multer';
import mongoose from 'mongoose';
import { Readable } from 'stream';
import PizZip from 'pizzip';
import { gfsBucket } from '../config/db.js';
import Template from '../models/Template.js';
import { seedDefaultTemplate } from '../services/seedService.js';

const router = express.Router();

// Use memory storage - we will pipe to GridFS manually
const storage = multer.memoryStorage();
const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
      cb(null, true);
    } else {
      cb(new Error('Only .docx files are allowed'), false);
    }
  },
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB
});

// POST /api/templates/upload
router.post('/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ detail: 'No .docx file uploaded' });
    
    const { name, modality } = req.body;
    if (!name || !modality) return res.status(400).json({ detail: 'Name and modality are required' });

    // Extract tags dynamically from the uploaded docx
    let detected_tags = [];
    try {
      const zip = new PizZip(req.file.buffer);
      if (zip.files['word/document.xml']) {
        const docXml = zip.files['word/document.xml'].asText();
        
        // Remove XML tags to parse raw text
        const plainText = docXml.replace(/<[^>]+>/g, '');
        
        // Match {tag_name} and {{tag_name}}
        const tagRegex = /\{{1,2}([a-zA-Z0-9_]+)\}{1,2}/g;
        const tagsSet = new Set();
        let match;
        while ((match = tagRegex.exec(plainText)) !== null) {
          tagsSet.add(match[1]);
        }
        detected_tags = Array.from(tagsSet);
      }
    } catch (err) {
      console.error('Error parsing docx tags, using fallback:', err);
    }

    if (detected_tags.length === 0) {
      detected_tags = ['patient_id', 'patient_name', 'age', 'sex', 'ref_by', 'reg_date', 'report_date', 'findings', 'impression'];
    }

    // Convert buffer to readable stream and upload to GridFS
    const readableStream = Readable.from(req.file.buffer);
    
    const uploadStream = gfsBucket.openUploadStream(req.file.originalname, {
      metadata: { contentType: req.file.mimetype }
    });

    await new Promise((resolve, reject) => {
      readableStream.pipe(uploadStream);
      uploadStream.on('finish', resolve);
      uploadStream.on('error', reject);
    });

    const fileId = uploadStream.id;

    const template = await Template.create({
      name,
      modality,
      file_id: fileId,
      detected_tags
    });

    res.status(201).json({
      message: 'Template uploaded successfully',
      template_id: template._id,
      file_id: fileId
    });

  } catch (error) {
    console.error('Template upload error:', error);
    res.status(500).json({ detail: error.message });
  }
});

// GET /api/templates
router.get('/', async (req, res) => {
  try {
    // Automatically seed/fix template if missing or corrupt
    await seedDefaultTemplate();
    
    const templates = await Template.find({});
    res.json(templates);
  } catch (error) {
    res.status(500).json({ detail: error.message });
  }
});

// DELETE /api/templates/:id
router.delete('/:id', async (req, res) => {
  try {
    const template = await Template.findOne({ _id: req.params.id });
    if (!template) return res.status(404).json({ detail: 'Template not found' });

    // Delete from GridFS
    await gfsBucket.delete(template.file_id);
    await template.deleteOne();

    res.json({ message: 'Template deleted successfully' });
  } catch (error) {
    res.status(500).json({ detail: error.message });
  }
});

export default router;
