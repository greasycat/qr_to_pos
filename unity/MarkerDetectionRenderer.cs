using Intel.RealSense;
using System.Collections;
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
    readonly Dictionary<string, Coroutine> pendingWallSpawnCoroutines = new Dictionary<string, Coroutine>();

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
        StopPendingWallSpawnCoroutines();
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
        var pendingWallSnapshots = new Dictionary<string, List<MarkerConstructionPointSnapshot>>();
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
                    TrackOrQueueWallPoint(detection, constructionAssignment, worldPosition, pendingWallSnapshots);
                    continue;
                }

                markerManager.TrackMarker(
                    markerParent,
                    worldPosition,
                    markerScale,
                    detection,
                    GetConstructionPrefab(constructionAssignment),
                    removeColorWhenUsingPrefab);
                continue;
            }

            markerManager.TrackMarker(
                markerParent,
                worldPosition,
                markerScale,
                detection,
                null,
                removeColorWhenUsingPrefab);
        }

        SchedulePendingWallSpawns(pendingWallSnapshots);
        if (constructionManager != null)
            constructionManager.RefreshTrackedConstructions(markerParent);
    }

    void TrackOrQueueWallPoint(
        MarkerDetection detection,
        MarkerConstructionAssignment constructionAssignment,
        Vector3 worldPosition,
        Dictionary<string, List<MarkerConstructionPointSnapshot>> pendingWallSnapshots)
    {
        if (constructionManager == null || constructionAssignment == null || constructionAssignment.Binding == null)
            return;

        MarkerConstructionPointSnapshot wallPointSnapshot;
        if (!TryCreateWallPointSnapshot(detection, constructionAssignment, worldPosition, out wallPointSnapshot))
            return;

        if (constructionManager.HasWallConstruction(constructionAssignment.BindingKey))
        {
            constructionManager.TrackWallPoint(
                wallPointSnapshot.Position,
                detection,
                constructionAssignment);
            return;
        }

        List<MarkerConstructionPointSnapshot> snapshots;
        if (!pendingWallSnapshots.TryGetValue(constructionAssignment.BindingKey, out snapshots))
        {
            snapshots = new List<MarkerConstructionPointSnapshot>();
            pendingWallSnapshots[constructionAssignment.BindingKey] = snapshots;
        }

        snapshots.Add(wallPointSnapshot);
    }

    bool TryCreateWallPointSnapshot(
        MarkerDetection detection,
        MarkerConstructionAssignment constructionAssignment,
        Vector3 worldPosition,
        out MarkerConstructionPointSnapshot wallPointSnapshot)
    {
        wallPointSnapshot = null;
        if (constructionAssignment == null || constructionAssignment.Binding == null)
            return false;

        string tagKey;
        if (!MarkerManager.TryGetTrackingKey(detection, out tagKey))
            return false;

        Vector3 groundedPosition = GetCachedGroundedWallPosition(tagKey, worldPosition);
        wallPointSnapshot = new MarkerConstructionPointSnapshot(
            constructionAssignment.BindingKey,
            constructionAssignment.Binding,
            constructionAssignment.DisplayName,
            tagKey,
            constructionAssignment.Order,
            groundedPosition);
        return true;
    }

    Vector3 GetCachedGroundedWallPosition(string tagKey, Vector3 worldPosition)
    {
        float currentTime = Time.time;
        float cacheDuration = Mathf.Max(0f, wallGroundRaycastCacheSeconds);

        CachedWallGroundPoint cachedPoint;
        if (wallGroundPointCache.TryGetValue(tagKey, out cachedPoint))
        {
            cachedPoint.LastSeenTime = currentTime;
            if (currentTime - cachedPoint.LastRaycastTime <= cacheDuration)
                return cachedPoint.Position;
        }

        float groundedSurfaceY;
        MarkerManager.ResolveGroundedCubePosition(
            markerParent,
            worldPosition,
            markerScale,
            out groundedSurfaceY);

        Vector3 groundedPosition = new Vector3(worldPosition.x, groundedSurfaceY, worldPosition.z);
        wallGroundPointCache[tagKey] = new CachedWallGroundPoint(groundedPosition, currentTime);
        return groundedPosition;
    }

    void SchedulePendingWallSpawns(Dictionary<string, List<MarkerConstructionPointSnapshot>> pendingWallSnapshots)
    {
        if (constructionManager == null || pendingWallSnapshots == null || pendingWallSnapshots.Count == 0)
            return;

        foreach (var entry in pendingWallSnapshots)
        {
            if (pendingWallSpawnCoroutines.ContainsKey(entry.Key) || constructionManager.HasWallConstruction(entry.Key))
                continue;

            List<MarkerConstructionPointSnapshot> snapshots = entry.Value;
            if (!HasCompleteWallSnapshot(snapshots))
                continue;

            Coroutine pendingCoroutine = StartCoroutine(SpawnWallFromSnapshotNextFrame(entry.Key, CloneWallSnapshots(snapshots)));
            pendingWallSpawnCoroutines[entry.Key] = pendingCoroutine;
        }
    }

    bool HasCompleteWallSnapshot(List<MarkerConstructionPointSnapshot> snapshots)
    {
        if (snapshots == null || snapshots.Count < 2)
            return false;

        snapshots.Sort(CompareWallSnapshots);
        MarkerConstructionBinding binding = snapshots[0].Binding;
        if (binding == null || binding.markerIndexes == null)
            return false;

        return snapshots.Count == binding.markerIndexes.Count;
    }

    IEnumerator SpawnWallFromSnapshotNextFrame(string bindingKey, List<MarkerConstructionPointSnapshot> pointSnapshots)
    {
        yield return new WaitForEndOfFrame();

        pendingWallSpawnCoroutines.Remove(bindingKey);
        if (constructionManager == null || constructionManager.HasWallConstruction(bindingKey))
            yield break;

        constructionManager.SpawnWallFromSnapshot(markerParent, pointSnapshots, Time.time);
    }

    void StopPendingWallSpawnCoroutines()
    {
        foreach (var entry in pendingWallSpawnCoroutines)
        {
            if (entry.Value != null)
                StopCoroutine(entry.Value);
        }

        pendingWallSpawnCoroutines.Clear();
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

    static int CompareWallSnapshots(MarkerConstructionPointSnapshot left, MarkerConstructionPointSnapshot right)
    {
        int orderComparison = left.Order.CompareTo(right.Order);
        if (orderComparison != 0)
            return orderComparison;

        return string.CompareOrdinal(left.TagKey, right.TagKey);
    }

    static List<MarkerConstructionPointSnapshot> CloneWallSnapshots(List<MarkerConstructionPointSnapshot> snapshots)
    {
        var clones = new List<MarkerConstructionPointSnapshot>(snapshots.Count);
        for (int i = 0; i < snapshots.Count; i++)
        {
            MarkerConstructionPointSnapshot snapshot = snapshots[i];
            clones.Add(new MarkerConstructionPointSnapshot(
                snapshot.BindingKey,
                snapshot.Binding,
                snapshot.DisplayName,
                snapshot.TagKey,
                snapshot.Order,
                snapshot.Position));
        }

        clones.Sort(CompareWallSnapshots);
        return clones;
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
        StopPendingWallSpawnCoroutines();
        wallGroundPointCache.Clear();
    }

    void HandleTerrainHeightmapChanged()
    {
        wallGroundPointCache.Clear();
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
        public readonly Vector3 Position;
        public readonly float LastRaycastTime;
        public float LastSeenTime;

        public CachedWallGroundPoint(Vector3 position, float currentTime)
        {
            Position = position;
            LastRaycastTime = currentTime;
            LastSeenTime = currentTime;
        }
    }
}
