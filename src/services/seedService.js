import fs from 'fs';
import path from 'path';
import { Readable } from 'stream';
import PizZip from 'pizzip';
import mongoose from 'mongoose';
import { getGfsBucket } from '../config/db.js';
import Template from '../models/Template.js';

// Helper: Read a GridFS file into a Buffer
async function readGridFSFile(fileId) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    const downloadStream = getGfsBucket().openDownloadStream(new mongoose.Types.ObjectId(fileId));
    downloadStream.on('data', chunk => chunks.push(chunk));
    downloadStream.on('end', () => resolve(Buffer.concat(chunks)));
    downloadStream.on('error', reject);
  });
}

export async function seedDefaultTemplate() {
  try {
    // Check if the default template exists
    const existing = await Template.findOne({ name: 'JP Diagnostics USG Template' });
    if (existing) {
      try {
        const buffer = await readGridFSFile(existing.file_id);
        // Test if this is a valid zip (docx) file
        new PizZip(buffer);
        console.log('✅ Default template exists and is valid. Skipping seed.');
        return;
      } catch (err) {
        console.warn('⚠️  Existing template is corrupt or empty. Cleaning up and re-seeding...', err.message);
        try {
          await getGfsBucket().delete(existing.file_id);
        } catch (e) {
          // File might not exist in GridFS
        }
        await existing.deleteOne();
      }
    }

    // Clean up any other corrupted/empty templates that might block report generation
    const allTemplates = await Template.find({});
    for (const temp of allTemplates) {
      try {
        const buffer = await readGridFSFile(temp.file_id);
        new PizZip(buffer);
      } catch (err) {
        console.warn(`⚠️ Removing corrupted template: ${temp.name} (${err.message})`);
        try {
          await getGfsBucket().delete(temp.file_id);
        } catch (e) {}
        await temp.deleteOne();
      }
    }

    // Use process.cwd() to resolve from the project root (where node src/server.js is run)
    const templatePath = path.resolve(process.cwd(), 'test_template.docx');
    if (!fs.existsSync(templatePath)) {
      console.error(`Cannot find default template at ${templatePath} to seed.`);
      return;
    }

    console.log(`🌱 Seeding default template from ${templatePath}`);
    const fileBuffer = fs.readFileSync(templatePath);

    // Extract tags
    let detected_tags = [];
    try {
      const zip = new PizZip(fileBuffer);
      if (zip.files['word/document.xml']) {
        const docXml = zip.files['word/document.xml'].asText();
        const plainText = docXml.replace(/<[^>]+>/g, '');
        const tagRegex = /\{{1,2}([a-zA-Z0-9_]+)\}{1,2}/g;
        const tagsSet = new Set();
        let match;
        while ((match = tagRegex.exec(plainText)) !== null) {
          tagsSet.add(match[1]);
        }
        detected_tags = Array.from(tagsSet);
      }
    } catch (err) {
      console.error('Error parsing docx tags during seed, using fallback:', err);
    }

    if (detected_tags.length === 0) {
      detected_tags = [
        'patient_id', 'patient_name', 'age', 'sex', 'ref_by', 'reg_date', 'report_date',
        'liver_finding', 'gallbladder_finding', 'pancreas_finding', 'spleen_finding',
        'kidneys_finding', 'urinary_bladder_finding', 'prostate_finding',
        'additional_finding', 'impression'
      ];
    }

    // Upload to GridFS
    const readableStream = Readable.from(fileBuffer);
    const uploadStream = getGfsBucket().openUploadStream('test_template.docx', {
      metadata: { contentType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' }
    });

    await new Promise((resolve, reject) => {
      readableStream.pipe(uploadStream);
      uploadStream.on('finish', resolve);
      uploadStream.on('error', reject);
    });

    const fileId = uploadStream.id;

    // Save Template record
    await Template.create({
      name: 'JP Diagnostics USG Template',
      modality: 'USG',
      file_id: fileId,
      detected_tags
    });

    console.log('✅ Default template seeded successfully!');
  } catch (err) {
    console.error('❌ Failed to seed default template:', err.message);
  }
}
