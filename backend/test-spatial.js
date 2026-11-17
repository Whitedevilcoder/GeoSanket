const mongoose = require("mongoose");
const dotenv = require("dotenv");
dotenv.config();
const Zone = require("./src/models/Zone");

async function testSpeed() {
  await mongoose.connect(process.env.MONGO_URI);
  console.log("Connected to MongoDB...");

  const lat = 28.61944; // Example coordinate
  const lng = 77.2425;

  const start = Date.now();
  
  // The exact query your app uses
  const zones = await Zone.find({
    isActive: true,
    centerPoint: {
      $near: {
        $geometry: { type: "Point", coordinates: [lng, lat] },
        $maxDistance: 1000,
      },
    },
  });

  const end = Date.now();
  console.log(`✅ Found ${zones.length} zones in ${end - start} ms`);
  process.exit(0);
}

testSpeed();