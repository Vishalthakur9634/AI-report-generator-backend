import express from 'express';

const router = express.Router();

// GET /api/templates
router.get('/', (req, res) => {
  // Return the hardcoded default template
  res.json([{
    _id: "default-template",
    name: "JP Diagnostics USG Template",
    modality: "USG",
    detected_tags: [
        'patient_id', 'patient_name', 'age', 'sex', 'ref_by', 'reg_date', 'report_date',
        'liver_finding', 'gallbladder_finding', 'pancreas_finding', 'spleen_finding',
        'kidneys_finding', 'urinary_bladder_finding', 'prostate_finding',
        'additional_finding', 'impression'
    ]
  }]);
});

// POST /api/templates/upload
router.post('/upload', (req, res) => {
  res.status(400).json({ detail: 'Template uploading is disabled in stateless mode. The default template is permanent.' });
});

// DELETE /api/templates/:id
router.delete('/:id', (req, res) => {
  res.status(400).json({ detail: 'Template deletion is disabled in stateless mode.' });
});

export default router;
