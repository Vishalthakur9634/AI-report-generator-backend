import mongoose from 'mongoose';

const reportSchema = mongoose.Schema({
  template_id: { type: String, required: true },
  status: { type: String, enum: ['generating', 'completed', 'failed'], default: 'completed' },
  file_id: { type: String, required: true }, // GridFS file ID of the final filled report
  ai_payload: { type: Object, required: true } // Store the JSON extracted by AI
}, {
  timestamps: true
});

const Report = mongoose.model('Report', reportSchema);
export default Report;
