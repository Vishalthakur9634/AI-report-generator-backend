import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';

import templateRoutes from './routes/templateRoutes.js';
import reportRoutes from './routes/reportRoutes.js';

dotenv.config();

const app = express();

// Middleware
app.use(cors({
  origin: true, // Dynamically reflect any origin
  credentials: true
}));
app.use(express.json());


app.use('/api/templates', templateRoutes);
app.use('/api/reports', reportRoutes);

app.get('/', (req, res) => {
  res.send('AI Radiology Node.js Backend Running');
});

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT} (Database Disabled)`);
});
setInterval(() => {}, 1000 * 60 * 60); // Keep alive
