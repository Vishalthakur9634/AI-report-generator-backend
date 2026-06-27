import mongoose from 'mongoose';

const templateSchema = mongoose.Schema({
  name: { type: String, required: true },
  modality: { type: String, required: true },
  file_id: { type: String, required: true }, // GridFS file ID as String to prevent CastError
  detected_tags: [{ type: String }]
}, {
  timestamps: true
});

const Template = mongoose.model('Template', templateSchema);
export default Template;
