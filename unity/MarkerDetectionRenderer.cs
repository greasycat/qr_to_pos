using Intel.RealSense;
using System.Collections.Generic;
using UnityEngine;

public class MarkerDetectionRenderer : MonoBehaviour
{
    public bool debugMode;
    public RsFrameProvider Source;

    public Stream _stream;
    public Format _format;
    public int _streamIndex;

    public FilterMode filterMode = FilterMode.Point;

    public string serverUrl = "ws://localhost:8765";
    public float sendInterval = 0.3f;

    public Terrain terrain;
    public Transform markerParent;
    public Vector3 markerScale = new Vector3(0.08f, 0.08f, 0.08f);
    public float markerVerticalOffset = 0.05f;
    public float wallGroundRaycastCacheSeconds = 0.15f;
    public float wallGroundMaxValidSurfaceY = 150f;
    public List<MarkerConstructionBinding> markerConstructionBindings = new List<MarkerConstructionBinding>();
    public bool removeColorWhenUsingPrefab = true;

    public bool showDebugBounds = true;
    public Color debugBoundsColor = Color.blue;
    public Vector3 debugBoundsScale = new Vector3(0.12f, 0.12f, 0.12f);
    public float debugBoundsY = 0.5f;

    public bool flipX;
    public bool flipZ = true;

    [SerializeField] int detectionCount;
    [SerializeField] int outOfBoundsConversionCount;

    readonly List<MarkerDetection> detections = new List<MarkerDetection>();
    readonly object detectionsLock = new object();

    MarkerFrameTextureSource frameTextureSource;
    MarkerDetectionWebSocketClient detectionClient;
    MarkerManager markerManager;
    MarkerConstructionManager constructionManager;
    Texture2D currentSourceTexture;
    int lastDebugTextureVersion = -1;
    readonly Dictionary<string, CachedWallGroundPoint> wallGroundPointCache = new Dictionary<string, CachedWallGroundPoint>();

    bool detectionsDirty;
    bool liveFrameSourceInitialized;
    bool missingSourceLogged;
    bool missingTerrainLogged;

    async void Start()
    {
        if (terrain == null)
            terrain = Terrain.activeTerrain;

        if (markerParent == null)
            markerParent = transform;

        markerManager = new MarkerManager();
        constructionManager = new MarkerConstructionManager();
        detectionsDirty = true;
        TerrainEvents.OnHeightmapChanged += HandleTerrainHeightmapChanged;

        if (!debugMode)
            TryInitializeLiveFrameSource();

        detectionClient = new MarkerDetectionWebSocketClient(serverUrl, HandleDetectionResponse);

        try
        {
            await detectionClient.ConnectAsync();
        }
        catch (System.Exception e)
        {
            Debug.LogException(e);
        }
    }

    void LateUpdate()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        if (detectionClient != null)
            detectionClient.DispatchMessageQueue();
#endif

        if (debugMode)
            PumpDebugTexture();
        else
            PumpLiveTexture();

        if (detectionsDirty)
            RefreshMarkers();

        if (markerManager != null)
            markerManager.PruneExpiredMarkers(Time.time);

        if (constructionManager != null)
            constructionManager.PruneExpiredConstructions(markerParent, Time.time);

        PruneWallGroundPointCache(Time.time);
    }

    async void OnDestroy()
    {
        if (frameTextureSource != null)
        {
            frameTextureSource.Dispose();
            frameTextureSource = null;
        }

        if (detectionClient != null)
        {
            try
            {
                await detectionClient.CloseAsync();
            }
            catch (System.Exception e)
            {
                Debug.LogException(e);
            }

            detectionClient = null;
        }

        TerrainEvents.OnHeightmapChanged -= HandleTerrainHeightmapChanged;
        wallGroundPointCache.Clear();

        if (markerManager != null)
            markerManager.ClearAll();

        if (constructionManager != null)
            constructionManager.ClearAll();
    }

    async void OnApplicationQuit()
    {
        if (detectionClient != null)
            await detectionClient.CloseAsync();
    }

    void OnSourceTextureUpdated(Texture2D sourceTexture)
    {
        currentSourceTexture = sourceTexture;
        if (detectionClient != null)
            detectionClient.TrySend(
                sourceTexture,
                sendInterval,
                debugMode
                    ? MarkerDetectionWebSocketClient.DetectionPayloadType.Detect
                    : MarkerDetectionWebSocketClient.DetectionPayloadType.DetectUnity);
    }

    void HandleDetectionResponse(DetectionResponse response)
    {
        lock (detectionsLock)
        {
            detections.Clear();
            if (response.detections != null)
                detections.AddRange(response.detections);

            detectionCount = detections.Count;
        }

        detectionsDirty = true;
    }

    void RefreshMarkers()
    {
        detectionsDirty = false;
        outOfBoundsConversionCount = 0;

        if (terrain == null)
        {
            markerManager.ClearDebugMarkers();
            if (!missingTerrainLogged)
            {
                Debug.LogWarning("MarkerDetectionRenderer: No terrain assigned, skipping marker placement.");
                missingTerrainLogged = true;
            }
            return;
        }

        missingTerrainLogged = false;

        MarkerTerrainMapper terrainMapper = CreateTerrainMapper();
        if (showDebugBounds || debugMode)
        {
            markerManager.BeginDebugMarkerRefresh();
            SpawnDebugBounds(terrainMapper);
            markerManager.EndDebugMarkerRefresh();
        }
        else
        {
            markerManager.ClearDebugMarkers();
        }

        if (currentSourceTexture == null)
            return;

        List<MarkerDetection> currentDetections;
        lock (detectionsLock)
        {
            currentDetections = new List<MarkerDetection>(detections);
        }

        Dictionary<int, MarkerConstructionAssignment> constructionLookup = BuildConstructionLookup();
        var latestDetectionsByTag = new Dictionary<string, MarkerDetection>(currentDetections.Count);
        for (int i = 0; i < currentDetections.Count; i++)
        {
            MarkerDetection detection = currentDetections[i];
            string tagKey;
            if (!MarkerManager.TryGetTrackingKey(detection, out tagKey))
            {
                Debug.LogWarningFormat(
                    "MarkerDetectionRenderer: Skipping detection without tag id. decoded='{0}'",
                    GetDetectionLabel(detection));
                continue;
            }

            if (latestDetectionsByTag.ContainsKey(tagKey))
            {
                Debug.LogWarningFormat(
                    "MarkerDetectionRenderer: Duplicate tag '{0}' detected in the same frame. Using the last detection.",
                    tagKey);
            }

            latestDetectionsByTag[tagKey] = detection;
        }

        foreach (var entry in latestDetectionsByTag)
        {
            MarkerDetection detection = entry.Value;
            Vector3 worldPosition;
            bool isOutOfBounds;
            if (!terrainMapper.TryGetMarkerPosition(detection, markerVerticalOffset, out worldPosition, out isOutOfBounds))
                continue;

            if (isOutOfBounds)
            {
                outOfBoundsConversionCount++;
                Debug.LogWarningFormat(
                    "MarkerDetectionRenderer: Out-of-bounds detection '{0}' depth_centroid_pct={1}",
                    GetDetectionLabel(detection),
                    GetDepthCentroidPercentageLabel(detection));
                continue;
            }

            MarkerConstructionAssignment constructionAssignment = GetConstructionAssignmentForDetection(detection, constructionLookup);
            if (constructionAssignment != null && constructionAssignment.Binding != null)
            {
                if (constructionAssignment.Binding.choice == MarkerConstructionChoice.Wall)
                {
                    TrackWallPoint(detection, constructionAssignment, worldPosition);
                    continue;
                }

                markerManager.TrackMarker(
                    markerParent,
                    worldPosition,
                    markerScale,
                    detection,
                    GetConstructionPrefab(constructionAssignment),
                    removeColorWhenUsingPrefab,
                    wallGroundRaycastCacheSeconds,
                    wallGroundMaxValidSurfaceY);
                continue;
            }

            markerManager.TrackMarker(
                markerParent,
                worldPosition,
                markerScale,
                detection,
                null,
                removeColorWhenUsingPrefab,
                wallGroundRaycastCacheSeconds,
                wallGroundMaxValidSurfaceY);
        }

        if (constructionManager != null)
            constructionManager.RefreshTrackedConstructions(markerParent);
    }

    void TrackWallPoint(
        MarkerDetection detection,
        MarkerConstructionAssignment constructionAssignment,
        Vector3 worldPosition)
    {
        if (constructionManager == null || constructionAssignment == null || constructionAssignment.Binding == null)
            return;

        Vector3 groundedPosition;
        if (!TryGetGroundedWallPosition(detection, worldPosition, out groundedPosition))
            return;

        constructionManager.TrackWallPoint(
            groundedPosition,
            detection,
            constructionAssignment);
    }

    bool TryGetGroundedWallPosition(
        MarkerDetection detection,
        Vector3 worldPosition,
        out Vector3 groundedPosition)
    {
        groundedPosition = worldPosition;

        string tagKey;
        if (!MarkerManager.TryGetTrackingKey(detection, out tagKey))
            return false;

        float currentTime = Time.time;
        float cacheDuration = Mathf.Max(0f, wallGroundRaycastCacheSeconds);
        bool hasRecentCache = false;

        CachedWallGroundPoint cachedPoint;
        if (wallGroundPointCache.TryGetValue(tagKey, out cachedPoint))
        {
            cachedPoint.LastSeenTime = currentTime;
            hasRecentCache = currentTime - cachedPoint.LastRaycastTime <= cacheDuration;
        }

        float groundedSurfaceY;
        MarkerManager.ResolveGroundedCubePosition(
            markerParent,
            worldPosition,
            markerScale,
            out groundedSurfaceY);

        if (IsValidWallGroundSurfaceY(groundedSurfaceY, worldPosition.y))
        {
            float cachedSurfaceY = hasRecentCache ? Mathf.Min(groundedSurfaceY, cachedPoint.SurfaceY) : groundedSurfaceY;
            groundedPosition = new Vector3(worldPosition.x, cachedSurfaceY, worldPosition.z);
            wallGroundPointCache[tagKey] = new CachedWallGroundPoint(cachedSurfaceY, currentTime);
            return true;
        }

        if (cachedPoint != null)
        {
            groundedPosition = new Vector3(worldPosition.x, cachedPoint.SurfaceY, worldPosition.z);
            return true;
        }

        return false;
    }

    void PruneWallGroundPointCache(float currentTime)
    {
        if (wallGroundPointCache.Count == 0)
            return;

        var expiredKeys = new List<string>();
        foreach (var entry in wallGroundPointCache)
        {
            if (entry.Value == null || currentTime - entry.Value.LastSeenTime > MarkerManager.MarkerLifetimeSeconds)
                expiredKeys.Add(entry.Key);
        }

        for (int i = 0; i < expiredKeys.Count; i++)
            wallGroundPointCache.Remove(expiredKeys[i]);
    }

    bool IsValidWallGroundSurfaceY(float surfaceY, float referenceY)
    {
        return !float.IsNaN(surfaceY)
            && !float.IsInfinity(surfaceY)
            && surfaceY - referenceY <= wallGroundMaxValidSurfaceY;
    }

    void SpawnDebugBounds(MarkerTerrainMapper terrainMapper)
    {
        List<MarkerDebugPlacement> markers = terrainMapper.GetDebugBounds(debugBoundsY);
        for (int i = 0; i < markers.Count; i++)
            markerManager.SpawnDebugMarker(markerParent, markers[i].WorldPosition, debugBoundsScale, debugBoundsColor, markers[i].Name);
    }

    MarkerTerrainMapper CreateTerrainMapper()
    {
        return new MarkerTerrainMapper(
            terrain,
            flipX,
            flipZ);
    }

    Dictionary<int, MarkerConstructionAssignment> BuildConstructionLookup()
    {
        if (markerConstructionBindings == null || markerConstructionBindings.Count == 0)
            return new Dictionary<int, MarkerConstructionAssignment>();

        var constructionLookup = new Dictionary<int, MarkerConstructionAssignment>();
        for (int i = 0; i < markerConstructionBindings.Count; i++)
        {
            MarkerConstructionBinding binding = markerConstructionBindings[i];
            if (binding == null || binding.markerIndexes == null || binding.markerIndexes.Count == 0)
                continue;

            string bindingKey = string.Format("construction_{0}", i);
            for (int order = 0; order < binding.markerIndexes.Count; order++)
            {
                int markerIndex = binding.markerIndexes[order];
                if (constructionLookup.ContainsKey(markerIndex))
                {
                    Debug.LogWarningFormat(
                        "MarkerDetectionRenderer: Marker index {0} is assigned to multiple construction bindings. Using the last binding '{1}'.",
                        markerIndex,
                        string.IsNullOrEmpty(binding.name) ? bindingKey : binding.name);
                }

                constructionLookup[markerIndex] = new MarkerConstructionAssignment(bindingKey, binding, markerIndex, order);
            }
        }

        return constructionLookup;
    }

    static MarkerConstructionAssignment GetConstructionAssignmentForDetection(
        MarkerDetection detection,
        Dictionary<int, MarkerConstructionAssignment> constructionLookup)
    {
        int markerIndex;
        if (!int.TryParse(detection.data, out markerIndex))
            return null;

        MarkerConstructionAssignment assignment;
        if (!constructionLookup.TryGetValue(markerIndex, out assignment))
            return null;

        return assignment;
    }

    static GameObject GetConstructionPrefab(MarkerConstructionAssignment constructionAssignment)
    {
        if (constructionAssignment == null || constructionAssignment.Binding == null)
            return null;

        if (constructionAssignment.Binding.choice != MarkerConstructionChoice.Prefab)
            return null;

        return constructionAssignment.Binding.prefab;
    }

    void PumpLiveTexture()
    {
        TryInitializeLiveFrameSource();

        if (frameTextureSource != null)
            frameTextureSource.PumpLatestFrame(filterMode, OnSourceTextureUpdated);
    }

    void PumpDebugTexture()
    {
        Texture2D debugTexture = MarkerDebugImageStore.SourceTexture;
        int debugTextureVersion = MarkerDebugImageStore.Version;

        if (debugTexture == null)
        {
            lastDebugTextureVersion = debugTextureVersion;
            if (currentSourceTexture != null)
            {
                currentSourceTexture = null;
                ClearDetections();
                detectionsDirty = true;
            }

            return;
        }

        lastDebugTextureVersion = debugTextureVersion;
        if (debugTexture != null)
            OnSourceTextureUpdated(debugTexture);
    }

    void TryInitializeLiveFrameSource()
    {
        if (liveFrameSourceInitialized)
            return;

        if (Source == null)
        {
            if (!missingSourceLogged)
            {
                Debug.LogError("MarkerDetectionRenderer: Source is not assigned.");
                missingSourceLogged = true;
            }
            return;
        }

        frameTextureSource = new MarkerFrameTextureSource(Source, _stream, _format, _streamIndex);
        frameTextureSource.Initialize();
        liveFrameSourceInitialized = true;
        missingSourceLogged = false;
    }

    void ClearDetections()
    {
        lock (detectionsLock)
        {
            detections.Clear();
            detectionCount = 0;
        }

        outOfBoundsConversionCount = 0;
        wallGroundPointCache.Clear();
        if (markerManager != null)
            markerManager.ClearGroundPlacementCache();
    }

    void HandleTerrainHeightmapChanged()
    {
        wallGroundPointCache.Clear();
        if (markerManager != null)
            markerManager.ClearGroundPlacementCache();
        detectionsDirty = true;
    }

    static string GetDetectionLabel(MarkerDetection detection)
    {
        if (!string.IsNullOrEmpty(detection.decoded))
            return detection.decoded;
        if (!string.IsNullOrEmpty(detection.data))
            return detection.data;

        return "Marker";
    }

    static string GetDepthCentroidPercentageLabel(MarkerDetection detection)
    {
        if (detection.depth_centroid_pct == null)
            return "<null>";
        if (detection.depth_centroid_pct.Length < 2)
            return "<invalid>";

        return string.Format(
            "({0:F4}, {1:F4})",
            detection.depth_centroid_pct[0],
            detection.depth_centroid_pct[1]);
    }

    sealed class CachedWallGroundPoint
    {
        public readonly float SurfaceY;
        public readonly float LastRaycastTime;
        public float LastSeenTime;

        public CachedWallGroundPoint(float surfaceY, float currentTime)
        {
            SurfaceY = surfaceY;
            LastRaycastTime = currentTime;
            LastSeenTime = currentTime;
        }
    }
}
