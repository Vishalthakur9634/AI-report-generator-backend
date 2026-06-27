import mongoose from 'mongoose';

const centerSchema = mongoose.Schema({
  name: { type: String, required: true },
  subscription_tier: { type: String, default: 'free' },
  is_active: { type: Boolean, default: true }
}, {
  timestamps: true
});

const Center = mongoose.model('Center', centerSchema);
export default Center;
