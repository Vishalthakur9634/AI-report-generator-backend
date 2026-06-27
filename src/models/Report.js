import mongoose from 'mongoose';

const reportSchema = mongoose.Schema({
  center_id: { type: mongoose.Schema.Types.ObjectId, ref: 'Center' },
  template_id: { type: mongoose.Schema.Types.ObjectId, ref: 'Template', required: true },
  created_by: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  status: { type: String, enum: ['generating', 'completed', 'failed'], default: 'completed' },
  file_id: { type: mongoose.Schema.Types.ObjectId, required: true }, // GridFS file ID of the final filled report
  ai_payload: { type: Object, required: true } // Store the JSON extracted by AI
}, {
  timestamps: true
});

const Report = mongoose.model('Report', reportSchema);
export default Report;
