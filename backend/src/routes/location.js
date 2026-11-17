const router = require("express").Router();
const { protect, asyncHandler } = require("../middleware/auth");
const { findZonesNearPoint } = require("../services/geoService");
const { 
  checkDedup, setDedup, getCachedZones, setCachedZones, getDwellStart, setDwellStart,
  getDailyNotificationCount, incrementDailyNotificationCount 
} = require("../services/redisService");
const { sendPushNotification } = require("../services/notificationService");

// POST /api/location/ping
router.post("/ping", protect, asyncHandler(async (req, res) => {
  const { lat, lng, zoneId, enteredZone } = req.body;
  const userId = req.user._id;
  const fcmToken = req.user.fcmToken;
  const lang = req.user.language || "hi";
  const results = [];

  // Direct zone entry from OS geofencing
  if (zoneId && enteredZone) {
    const Zone = require("../models/Zone");
    const zone = await Zone.findById(zoneId).populate("project");
    if (zone && zone.project) {
      const duped = await checkDedup(userId, zoneId);
      if (!duped) {
        // TIER 3: Check Fatigue Limit
        const dailyCount = await getDailyNotificationCount(userId);
        
        // HACK FOR SCREENSHOT: Change back to >= 5 later!
        if (dailyCount >= 5) {
          results.push({ zoneId, status: "suppressed_fatigue" });
          return res.json({ success: true, notifications: results });
        }

        // Try FCM if token exists
        if (fcmToken) {
          const r = await sendPushNotification({ userId, fcmToken, lang, project: zone.project, zone, location: { lat, lng } });
          if (r.success) {
            await setDedup(userId, zoneId);
            await incrementDailyNotificationCount(userId); 
          }
          results.push({ zoneId, status: r.success ? "notified" : r.reason, project: zone.project });
        } else {
          // No FCM token (browser demo)
          const NotificationLog = require("../models/NotificationLog");
          await NotificationLog.create({ user: userId, zone: zone._id, project: zone.project._id, status: "sent", channel: "browser", location: { lat, lng } });
          await setDedup(userId, zoneId);
          await incrementDailyNotificationCount(userId); 
          results.push({ zoneId, status: "notified", project: zone.project });
        }
      } else {
        results.push({ zoneId, status: "deduped", project: zone.project });
      }
    }
    return res.json({ success: true, notifications: results });
  }

  // Coordinate-based ping (Socket.io fallback)
  if (!lat || !lng) return res.status(400).json({ success: false, error: "lat/lng required" });

  let zones = await getCachedZones(lat, lng);
  if (!zones) {
    zones = await findZonesNearPoint(lat, lng);
    await setCachedZones(lat, lng, zones);
  }

  for (const zone of zones) {
    if (!zone.project) continue;
    
    // TIER 1: Dedup
    const duped = await checkDedup(userId, zone._id);
    if (duped) { results.push({ zoneId: zone._id, status: "deduped", message: "Already notified in last 24hrs" }); continue; }

    // TIER 2: Dwell time
    const dwellStart = await getDwellStart(userId, zone._id);
    if (!dwellStart) { await setDwellStart(userId, zone._id); results.push({ zoneId: zone._id, status: "dwell_started" }); continue; }
    const dwellSecs = (Date.now() - parseInt(dwellStart)) / 1000;
    if (dwellSecs < zone.dwellTimeSeconds) { results.push({ zoneId: zone._id, status: `dwelling_${Math.floor(dwellSecs)}s` }); continue; }

    // TIER 3: Fatigue Limit
    const dailyCount = await getDailyNotificationCount(userId);
    
    // HACK FOR SCREENSHOT: Change back to >= 5 later!
    if (dailyCount >= 5) {
      results.push({ zoneId: zone._id, status: "suppressed_fatigue" });
      continue; 
    }

    if (fcmToken) {
      const r = await sendPushNotification({ userId, fcmToken, lang, project: zone.project, zone, location: { lat, lng } });
      if (r.success) {
        await setDedup(userId, zone._id);
        await incrementDailyNotificationCount(userId);
      }
      results.push({ zoneId: zone._id, status: r.success ? "notified" : r.reason, project: zone.project });
    } else {
      const NotificationLog = require("../models/NotificationLog");
      await NotificationLog.create({ user: userId, zone: zone._id, project: zone.project._id, status: "sent", channel: "browser-gps", location: { lat, lng } });
      await setDedup(userId, zone._id);
      await incrementDailyNotificationCount(userId);
      results.push({ zoneId: zone._id, status: "notified", project: zone.project });
    }
  }

  res.json({ success: true, notifications: results, zonesChecked: zones.length });
}));

// GET /api/location/nearby
router.get("/nearby", protect, asyncHandler(async (req, res) => {
  const { lat, lng } = req.query;
  const zones = await findZonesNearPoint(parseFloat(lat), parseFloat(lng));
  res.json({ success: true, zones });
}));

// POST /api/v1/fence-check
router.post("/v1/fence-check", asyncHandler(async (req, res) => {
  const { lat, lng, deviceId, appId, lang = "hi" } = req.body;

  if (!lat || !lng || !deviceId) {
    return res.status(400).json({ success: false, error: "lat, lng and deviceId are required" });
  }

  const User = require("../models/User");
  let user = await User.findOne({ deviceId });
  if (!user) {
    user = await User.create({ deviceId, isAnonymous: true, language: lang });
  }

  const Zone = require("../models/Zone");
  const zones = await Zone.find({
    isActive: true,
    centerPoint: {
      $near: {
        $geometry: { type: "Point", coordinates: [parseFloat(lng), parseFloat(lat)] },
        $maxDistance: 1000
      }
    }
  }).populate("project");

  if (zones.length === 0) {
    return res.json({ success: true, shouldNotify: false, message: "No zones nearby" });
  }

  const notifications = [];

  for (const zone of zones) {
    if (!zone.project) continue;

    const duped = await checkDedup(user._id, zone._id);
    if (duped) continue;

    const dwellStart = await getDwellStart(user._id, zone._id);
    if (!dwellStart) {
      await setDwellStart(user._id, zone._id);
      continue;
    }
    const dwellSecs = (Date.now() - parseInt(dwellStart)) / 1000;
    if (dwellSecs < zone.dwellTimeSeconds) continue;

    const dailyCount = await getDailyNotificationCount(user._id);
    
    // HACK FOR SCREENSHOT: Change back to >= 5 later!
    if (dailyCount >= 5) continue;

    const NotificationLog = require("../models/NotificationLog");
    await NotificationLog.create({
      user: user._id, zone: zone._id,
      project: zone.project._id,
      status: "sent", channel: appId || "api",
      location: { lat, lng }
    });
    await setDedup(user._id, zone._id);
    await incrementDailyNotificationCount(user._id);

    const title = zone.project.notificationTitle?.[lang] || zone.project.notificationTitle?.en || zone.project.name?.en;
    const body = zone.project.notificationBody?.[lang] || zone.project.notificationBody?.en || `${zone.project.name?.en} · ₹${((zone.project.budget?.sanctioned || 0) / 1e7).toFixed(0)}Cr · ${zone.project.completionPercentage}% complete`;

    notifications.push({
      zoneId: zone._id, zoneName: zone.name, title, body,
      projectId: zone.project._id, projectName: zone.project.name?.en,
      budget: zone.project.budget, completionPercentage: zone.project.completionPercentage,
      leader: zone.project.leader, scheme: zone.project.scheme, department: zone.project.department,
    });
  }

  if (notifications.length === 0) {
    return res.json({ success: true, shouldNotify: false, message: "Dedup active, dwell not reached, or fatigue limit hit." });
  }

  res.json({ success: true, shouldNotify: true, count: notifications.length, notifications });
}));

module.exports = router;