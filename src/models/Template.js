import mongoose from 'mongoose';

const templateSchema = mongoose.Schema({
  center_id: { type: mongoose.Schema.Types.ObjectId, ref: 'Center' },
  name: { type: String, required: true },
  modality: { type: String, required: true },
  file_id: { type: mongoose.Schema.Types.ObjectId, required: true }, // GridFS file ID
  detected_tags: [{ type: String }],
  uploaded_by: { type: mongoose.Schema.Types.ObjectId, ref: 'User' }
}, {
  timestamps: true
});

const Template = mongoose.model('Template', templateSchema);
export default Template;
