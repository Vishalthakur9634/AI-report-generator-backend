import express from 'express';
import PizZip from 'pizzip';
import Docxtemplater from 'docxtemplater';
import { Readable } from 'stream';
import mongoose from 'mongoose';
import { gfsBucket } from '../config/db.js';
import Template from '../models/Template.js';
import Report from '../models/Report.js';
import { generateReportData } from '../services/aiService.js';

const router = express.Router();

// Helper: Read a GridFS file into a Buffer
async function readGridFSFile(fileId) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    const downloadStream = gfsBucket.openDownloadStream(new mongoose.Types.ObjectId(fileId));
    downloadStream.on('data', chunk => chunks.push(chunk));
    downloadStream.on('end', () => resolve(Buffer.concat(chunks)));
    downloadStream.on('error', reject);
  });
}

// Helper: Save a Buffer to GridFS and return the new file ID
async function saveToGridFS(buffer, filename) {
  return new Promise((resolve, reject) => {
    const readableStream = Readable.from(buffer);
    const uploadStream = gfsBucket.openUploadStream(filename, {
      metadata: { contentType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' }
    });
    readableStream.pipe(uploadStream);
    uploadStream.on('finish', () => resolve(uploadStream.id));
    uploadStream.on('error', reject);
  });
}

// POST /api/reports/generate
router.post('/generate', async (req, res) => {
  const { template_id, transcript } = req.body;

  if (!template_id || !transcript) {
    return res.status(400).json({ detail: 'template_id and transcript are required' });
  }

  try {
    // 1. Fetch the template (with center_id check for security)
    const templateDoc = await Template.findOne({
      _id: template_id
    });
    if (!templateDoc) return res.status(404).json({ detail: 'Template not found' });

    // 2. Call HuggingFace AI to get structured JSON
    console.log('Calling HuggingFace AI...');
    const reportData = await generateReportData(
      transcript,
      templateDoc.detected_tags,
      templateDoc.modality
    );
    console.log('AI data received:', reportData);

    // 3. Fetch the .docx from GridFS
    const docxBuffer = await readGridFSFile(templateDoc.file_id);

    // 4. Use docxtemplater to inject AI data into the .docx
    const zip = new PizZip(docxBuffer);
    const doc = new Docxtemplater(zip, {
      paragraphLoop: true,
      linebreaks: true,
      // Error handler — if a tag is missing in the data, fill with empty string
      nullGetter: () => ''
    });
    doc.render(reportData);
    const filledBuffer = doc.getZip().generate({ type: 'nodebuffer' });

    // 5. Save the filled .docx to GridFS
    const fileName = `Report_Guest_${Date.now()}.docx`;
    const finalFileId = await saveToGridFS(filledBuffer, fileName);

    // 6. Save the report audit record to MongoDB
    const report = await Report.create({
      template_id: templateDoc._id,
      status: 'completed',
      file_id: finalFileId,
      ai_payload: reportData
    });

    res.status(201).json({
      message: 'Report generated successfully',
      report_id: report._id,
      file_id: finalFileId.toString(),
      preview_data: reportData
    });

  } catch (error) {
    console.error('Report generation error:', error);
    if (error.message === 'GROQ_RATE_LIMIT_EXCEEDED') {
      return res.status(429).json({ detail: 'GROQ_RATE_LIMIT_EXCEEDED' });
    }
    res.status(500).json({ detail: error.message });
  }
});

// GET /api/reports/download/:fileId  — Download the filled .docx
router.get('/download/:fileId', async (req, res) => {
  try {
    const fileId = new mongoose.Types.ObjectId(req.params.fileId);

    res.set({
      'Content-Type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'Content-Disposition': `attachment; filename="report.docx"`
    });

    const downloadStream = gfsBucket.openDownloadStream(fileId);
    downloadStream.pipe(res);
    downloadStream.on('error', () => res.status(404).json({ detail: 'File not found' }));
  } catch (error) {
    res.status(500).json({ detail: error.message });
  }
});

// GET /api/reports — Fetch all reports for this center
router.get('/', async (req, res) => {
  try {
    const reports = await Report.find({})
      .sort({ createdAt: -1 })
      .populate('template_id', 'name modality');
    res.json(reports);
  } catch (error) {
    res.status(500).json({ detail: error.message });
  }
});

export default router;
