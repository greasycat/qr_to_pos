using Intel.RealSense;
using System.Collections.Generic;
using UnityEngine;

public class MarkerDetectionRenderer : MonoBehaviour
{
    public enum DebugMode { None, MockServer, SingleImage }

    public DebugMode debugMode;
    public RsFrameProvider Source;

    public Stream _stream;
    public Format _format;
    public int _streamIndex;

    public FilterMode filterMode = FilterMode.Point;

    public string serverUrl = "ws://localhost:8765";
    public float sendInterval = 0.3f;

    public Terrain terrain;
    public Transform markerParent;
    public Vector3 markerScale = new Vector3(30f, 30f, 30f);
    public float markerVerticalOffset = 0.05f;
    public List<MarkerConstructionBinding> markerConstructionBindings = new List<MarkerConstructionBinding>();
    public bool removeColorWhenUsingPrefab = true;

    public bool showDebugBounds = true;
    public Color debugBoundsColor = Color.blue;
    public Vector3 debugBoundsScale = new Vector3(30f, 30f, 30f);
    public float debugBoundsY = 0.5f;

    public bool flipX;
    public bool flipZ;

    [SerializeField] int detectionCount;
    [SerializeField] int outOfBoundsConversionCount;

    readonly List<MarkerDetection> detections = new List<MarkerDetection>();
    readonly object detectionsLock = new object();

    Texture2D mockServerDummyTexture;
    MarkerFrameTextureSource frameTextureSource;
    MarkerDetectionWebSocketClient detectionClient;
    MarkerManager markerManager;
    MarkerConstructionManager constructionManager;
    Texture2D currentSourceTexture;
    int lastDebugTextureVersion = -1;

    bool detectionsDirty;
    bool liveFrameSourceInitialized;
    bool missingSourceLogged;
    bool missingTerrainLogged;

    async void Start()
    {
        if (terrain == null)
            terrain = Terrain.activeTerrain ?? Object.FindObjectOfType<Terrain>();

        if (markerParent == null)
            markerParent = transform;

        markerManager = new MarkerManager();
        constructionManager = new MarkerConstructionManager();
        detectionsDirty = true;

        if (debugMode == DebugMode.None)
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

        if (debugMode == DebugMode.MockServer)
            PumpMockServerTexture();
        else if (debugMode == DebugMode.SingleImage)
            PumpSingleImageTexture();
        else
            PumpLiveTexture();

        if (detectionsDirty)
            RefreshMarkers();

        if (markerManager != null)
            markerManager.PruneExpiredMarkers(Time.time);
    }

    async void OnDestroy()
    {
        if (mockServerDummyTexture != null)
        {
            UnityEngine.Object.Destroy(mockServerDummyTexture);
            mockServerDummyTexture = null;
        }

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
                debugMode == DebugMode.SingleImage
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
            if (constructionManager != null)
                constructionManager.ClearWalls();
            if (!missingTerrainLogged)
            {
                Debug.LogWarning("MarkerDetectionRenderer: No terrain assigned, skipping marker placement.");
                missingTerrainLogged = true;
            }
            return;
        }

        missingTerrainLogged = false;

        MarkerTerrainMapper terrainMapper = CreateTerrainMapper();
        if (showDebugBounds || debugMode != DebugMode.None)
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
        {
            if (constructionManager != null)
                constructionManager.ClearWalls();
            return;
        }

        List<MarkerDetection> currentDetections;
        lock (detectionsLock)
        {
            currentDetections = new List<MarkerDetection>(detections);
        }

        Dictionary<int, MarkerConstructionAssignment> constructionLookup = BuildConstructionLookup();
        Dictionary<string, PendingWallBinding> pendingWalls = BuildPendingWallBindings();
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
                    TrackWallPoint(constructionAssignment, worldPosition, pendingWalls);
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

        if (constructionManager != null)
            ApplyWallBindings(pendingWalls);
    }

    void TrackWallPoint(
        MarkerConstructionAssignment constructionAssignment,
        Vector3 worldPosition,
        Dictionary<string, PendingWallBinding> pendingWalls)
    {
        if (constructionAssignment == null || constructionAssignment.Binding == null || pendingWalls == null)
            return;

        PendingWallBinding pendingWall;
        if (!pendingWalls.TryGetValue(constructionAssignment.BindingKey, out pendingWall) || pendingWall == null)
            return;

        if (constructionAssignment.Order < 0 || constructionAssignment.Order >= pendingWall.Positions.Length)
            return;

        Vector3 groundedPosition = worldPosition;
        if (!TryGetGroundedWallPosition(worldPosition, out groundedPosition))
            return;

        pendingWall.Positions[constructionAssignment.Order] = groundedPosition;
        pendingWall.HasPositions[constructionAssignment.Order] = true;
    }

    bool TryGetGroundedWallPosition(
        Vector3 worldPosition,
        out Vector3 groundedPosition)
    {
        groundedPosition = worldPosition;

        if (terrain == null)
            return false;

        float surfaceY = terrain.SampleHeight(worldPosition) + terrain.GetPosition().y;
        if (float.IsNaN(surfaceY) || float.IsInfinity(surfaceY))
            return false;

        groundedPosition = new Vector3(worldPosition.x, surfaceY, worldPosition.z);
        return true;
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

    Dictionary<string, PendingWallBinding> BuildPendingWallBindings()
    {
        var pendingWalls = new Dictionary<string, PendingWallBinding>();
        if (markerConstructionBindings == null || markerConstructionBindings.Count == 0)
            return pendingWalls;

        for (int i = 0; i < markerConstructionBindings.Count; i++)
        {
            MarkerConstructionBinding binding = markerConstructionBindings[i];
            if (binding == null
                || binding.choice != MarkerConstructionChoice.Wall
                || binding.markerIndexes == null
                || binding.markerIndexes.Count == 0)
            {
                continue;
            }

            string bindingKey = BuildBindingKey(i);
            pendingWalls[bindingKey] = new PendingWallBinding(bindingKey, binding);
        }

        return pendingWalls;
    }

    void ApplyWallBindings(Dictionary<string, PendingWallBinding> pendingWalls)
    {
        if (constructionManager == null)
            return;

        constructionManager.BeginWallRefresh();

        if (pendingWalls != null)
        {
            foreach (var entry in pendingWalls)
            {
                PendingWallBinding pendingWall = entry.Value;
                if (pendingWall == null || !pendingWall.IsComplete)
                    continue;

                constructionManager.UpdateWall(
                    markerParent,
                    pendingWall.BindingKey,
                    pendingWall.DisplayName,
                    pendingWall.Binding.wall,
                    pendingWall.Positions);
            }
        }

        constructionManager.EndWallRefresh();
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

            string bindingKey = BuildBindingKey(i);
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

    static string BuildBindingKey(int bindingIndex)
    {
        return string.Format("construction_{0}", bindingIndex);
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

    void PumpMockServerTexture()
    {
        if (mockServerDummyTexture == null)
            mockServerDummyTexture = new Texture2D(1, 1);

        OnSourceTextureUpdated(mockServerDummyTexture);
    }

    void PumpSingleImageTexture()
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

    sealed class PendingWallBinding
    {
        public readonly string BindingKey;
        public readonly MarkerConstructionBinding Binding;
        public readonly Vector3[] Positions;
        public readonly bool[] HasPositions;

        public PendingWallBinding(string bindingKey, MarkerConstructionBinding binding)
        {
            BindingKey = bindingKey;
            Binding = binding;
            int markerCount = binding != null && binding.markerIndexes != null ? binding.markerIndexes.Count : 0;
            Positions = new Vector3[markerCount];
            HasPositions = new bool[markerCount];
        }

        public string DisplayName
        {
            get
            {
                if (Binding != null && !string.IsNullOrEmpty(Binding.name))
                    return Binding.name;

                return BindingKey;
            }
        }

        public bool IsComplete
        {
            get
            {
                if (Positions == null || Positions.Length < 2 || HasPositions == null || HasPositions.Length != Positions.Length)
                    return false;

                for (int i = 0; i < HasPositions.Length; i++)
                {
                    if (!HasPositions[i])
                        return false;
                }

                return true;
            }
        }
    }
}
