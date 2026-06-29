import mongoose from 'mongoose';

let bucket;

const connectDB = async () => {
  try {
    const conn = await mongoose.connect(process.env.MONGODB_URI, {
      serverSelectionTimeoutMS: 10000,
      socketTimeoutMS: 45000,
      maxPoolSize: 10,
      tlsAllowInvalidCertificates: true
    });

    console.log(`MongoDB Connected: ${conn.connection.host}`);

    bucket = new mongoose.mongo.GridFSBucket(conn.connection.db, {
      bucketName: 'uploads'
    });
    console.log('GridFS initialized');

    mongoose.connection.on('disconnected', () => {
      console.warn('⚠️  MongoDB disconnected! Attempting to reconnect...');
    });
    mongoose.connection.on('reconnected', () => {
      console.log('✅ MongoDB reconnected!');
    });
    mongoose.connection.on('error', (err) => {
      console.error('MongoDB connection error:', err.message);
    });

  } catch (error) {
    console.error(`MongoDB connection failed: ${error.message}`);
    console.error('Make sure your MongoDB Atlas IP whitelist includes 0.0.0.0/0 (Allow All)');
  }
};

export const getGfsBucket = () => bucket;
export default connectDB;
