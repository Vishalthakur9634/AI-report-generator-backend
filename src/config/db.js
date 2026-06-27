import mongoose from 'mongoose';

// We will store the global gridfs bucket here so we can access it from routes
export let gfsBucket;

const connectDB = async () => {
  try {
    const conn = await mongoose.connect(process.env.MONGODB_URI, {
      serverSelectionTimeoutMS: 10000, // Timeout after 10s if can't connect
      socketTimeoutMS: 45000,          // Close socket if no response in 45s
      maxPoolSize: 10,                 // Maintain up to 10 socket connections
      tlsAllowInvalidCertificates: true // Allow invalid/self-signed certs (resolves local SSL/TLS chain issues)
    });

    console.log(`MongoDB Connected: ${conn.connection.host}`);

    // Initialize GridFSBucket
    gfsBucket = new mongoose.mongo.GridFSBucket(conn.connection.db, {
      bucketName: 'uploads'
    });
    console.log('GridFS initialized');

    // Handle disconnection events gracefully
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

export default connectDB;

