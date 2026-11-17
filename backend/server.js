const http    = require("http");
const app     = require("./src/app");
const connectDB = require("./src/config/db");
const { initSocket } = require("./src/socket/locationSocket");
const { getRedis } = require("./src/config/redis");
getRedis();

const PORT = process.env.PORT || 5000;
connectDB();

const server = http.createServer(app);
initSocket(server);

server.listen(PORT, () => {
  console.log(`🚀 GeoSanket backend running on port ${PORT}`);
});