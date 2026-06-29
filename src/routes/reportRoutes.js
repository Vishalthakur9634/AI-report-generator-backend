import express from 'express';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import Docxtemplater from 'docxtemplater';
import PizZip from 'pizzip';
import { generateReportData } from '../services/aiService.js';

const router = express.Router();

// In-memory cache for generated reports.
// Maps a unique file_id (string) to a Buffer.
const reportCache = new Map();

// POST /api/reports/generate
router.post('/generate', async (req, res) => {
  try {
    const { template_id, transcript } = req.body;
    
    if (!transcript) {
      return res.status(400).json({ detail: 'Transcript is required' });
    }
    
    // In stateless mode, we only support the hardcoded template.
    if (template_id !== 'default-template') {
      return res.status(400).json({ detail: 'Invalid template_id. Only default-template is supported.' });
    }

    // 1. Read the default template from local disk
    const templatePath = path.resolve(process.cwd(), 'test_template.docx');
    if (!fs.existsSync(templatePath)) {
      return res.status(500).json({ detail: 'Server Error: test_template.docx not found on disk.' });
    }
    const templateBuffer = fs.readFileSync(templatePath);

    // Extract tags from template for AI processing
    const detected_tags = [
      'patient_id', 'patient_name', 'age', 'sex', 'ref_by', 'reg_date', 'report_date',
      'liver_finding', 'gallbladder_finding', 'pancreas_finding', 'spleen_finding',
      'kidneys_finding', 'urinary_bladder_finding', 'prostate_finding',
      'additional_finding', 'impression'
    ];

    // 2. Generate JSON mapping via Groq AI
    const mappedData = await generateReportData(transcript, detected_tags, 'USG');

    // 3. Process DOCX with Docxtemplater
    const zip = new PizZip(templateBuffer);
    const doc = new Docxtemplater(zip, {
      paragraphLoop: true,
      linebreaks: true,
      nullGetter() { return ""; } // Replace undefined/null with empty string
    });

    doc.render(mappedData);
    
    // 4. Generate final Buffer
    const generatedBuffer = doc.getZip().generate({
      type: 'nodebuffer',
      compression: 'DEFLATE'
    });

    // 5. Store in memory cache
    const file_id = crypto.randomUUID();
    reportCache.set(file_id, generatedBuffer);
    
    // Automatically delete from cache after 5 minutes to prevent memory leaks
    setTimeout(() => {
      reportCache.delete(file_id);
    }, 5 * 60 * 1000);

    res.json({
      message: 'Report generated successfully',
      report_id: 'in-memory',
      file_id: file_id,
      preview_data: mappedData
    });

  } catch (error) {
    console.error('Report Generation Error:', error);
    if (error.message === 'GROQ_RATE_LIMIT_EXCEEDED') {
      return res.status(429).json({ detail: 'GROQ_RATE_LIMIT_EXCEEDED' });
    }
    res.status(500).json({ detail: error.message || 'Failed to generate report' });
  }
});

// GET /api/reports/download/:fileId
router.get('/download/:fileId', async (req, res) => {
  try {
    const { fileId } = req.params;
    
    if (!reportCache.has(fileId)) {
      return res.status(404).json({ detail: 'Report expired or not found. Please generate again.' });
    }

    const buffer = reportCache.get(fileId);
    
    res.set({
      'Content-Type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'Content-Disposition': 'attachment; filename="generated_report.docx"'
    });
    
    res.send(buffer);
    
    // Once downloaded, we can clear it from memory to save RAM
    reportCache.delete(fileId);
    
  } catch (error) {
    console.error('Download error:', error);
    res.status(500).json({ detail: 'Failed to download report' });
  }
});

export default router;
