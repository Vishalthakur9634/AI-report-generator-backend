import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import connectDB from './config/db.js';
import templateRoutes from './routes/templateRoutes.js';
import reportRoutes from './routes/reportRoutes.js';

dotenv.config();

const app = express();

// Middleware
const frontendUrl = (process.env.FRONTEND_URL || 'http://localhost:5173').replace(/\/$/, '');
app.use(cors({
  origin: function (origin, callback) {
    // Allow if no origin (e.g. mobile apps, curl), or if it matches frontendUrl, or if it's a local Vite port
    if (!origin || origin === frontendUrl || origin.startsWith('http://localhost:517')) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true
}));
app.use(express.json());


app.use('/api/templates', templateRoutes);
app.use('/api/reports', reportRoutes);

app.get('/', (req, res) => {
  res.send('AI Radiology Node.js Backend Running');
});

const PORT = process.env.PORT || 5000;

// Connect to MongoDB, then start server
connectDB().then(() => {
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
}).catch(err => {
  console.error("Failed to connect to MongoDB", err);
});
