using Intel.RealSense;
using System.Collections.Generic;
using UnityEngine;

public class QRDetectionRenderer : MonoBehaviour
{
    [Header("Input")]
    public bool debugMode;
    public RsFrameProvider Source;

    public Stream _stream;
    public Format _format;
    public int _streamIndex;

    public FilterMode filterMode = FilterMode.Point;

    [Header("WebSocket")]
    public string serverUrl = "ws://localhost:8765";
    public float sendInterval = 0.3f;
    public bool flipHorizontallyBeforeDetection = true;
    public bool flipVerticallyBeforeDetection;

    [Header("Marker Placement")]
    public Terrain terrain;
    public Transform markerParent;
    public Vector3 markerScale = new Vector3(0.08f, 0.08f, 0.08f);
    public Color markerColor = Color.green;
    public float markerVerticalOffset = 0.05f;

    [Header("Marker Fall")]
    public bool enableMarkerFall = true;
    public float markerFallSpawnHeight = 0.4f;
    public float markerFallAcceleration = 4f;
    public float markerMaxFallSpeed = 2.5f;

    [Header("Spawn Filter")]
    public float markerRespawnDelaySeconds = 5f;
    public float markerRespawnDistance = 10f;

    [Header("Debug Bounds")]
    public bool showDebugBounds = true;
    public Color debugBoundsColor = Color.blue;
    public Vector3 debugBoundsScale = new Vector3(0.12f, 0.12f, 0.12f);
    public float debugBoundsY = 0.5f;

    public bool flipX;
    public bool flipZ = true;

    [Header("Runtime Stats")]
    [SerializeField] int detectionCount;
    [SerializeField] int outOfBoundsConversionCount;

    readonly List<QRDetection> detections = new List<QRDetection>();
    readonly object detectionsLock = new object();

    QRFrameTextureSource frameTextureSource;
    QRDetectionWebSocketClient detectionClient;
    QRMarkerManager markerManager;
    Texture2D currentSourceTexture;
    int lastDebugTextureVersion = -1;

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

        markerManager = new QRMarkerManager();
        detectionsDirty = true;
        TerrainEvents.OnHeightmapChanged += HandleTerrainHeightmapChanged;

        if (!debugMode)
            TryInitializeLiveFrameSource();

        detectionClient = new QRDetectionWebSocketClient(serverUrl, HandleDetectionResponse);

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

        if (markerManager != null)
            markerManager.ClearAll();
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
            detectionClient.TrySend(sourceTexture, sendInterval);
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
                Debug.LogWarning("QRDetectionRenderer: No terrain assigned, skipping marker placement.");
                missingTerrainLogged = true;
            }
            return;
        }

        missingTerrainLogged = false;

        QRTerrainMapper terrainMapper = CreateTerrainMapper();
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

        List<QRDetection> currentDetections;
        lock (detectionsLock)
        {
            currentDetections = new List<QRDetection>(detections);
        }

        for (int i = 0; i < currentDetections.Count; i++)
        {
            QRDetection detection = currentDetections[i];
            Vector3 worldPosition;
            bool isOutOfBounds;
            if (!terrainMapper.TryGetMarkerPosition(detection, markerVerticalOffset, out worldPosition, out isOutOfBounds))
                continue;

            if (isOutOfBounds)
            {
                outOfBoundsConversionCount++;
                Debug.LogWarningFormat(
                    "QRDetectionRenderer: Out-of-bounds detection '{0}' depth_centroid_pct={1}",
                    GetDetectionLabel(detection),
                    GetDepthCentroidPercentageLabel(detection));
                continue;
            }

            markerManager.SpawnMarker(
                markerParent,
                worldPosition,
                markerScale,
                markerColor,
                detection,
                i,
                enableMarkerFall,
                markerFallSpawnHeight,
                markerFallAcceleration,
                markerMaxFallSpeed,
                markerRespawnDelaySeconds,
                markerRespawnDistance);
        }
    }

    void SpawnDebugBounds(QRTerrainMapper terrainMapper)
    {
        List<QRDebugMarkerPlacement> markers = terrainMapper.GetDebugBounds(debugBoundsY);
        for (int i = 0; i < markers.Count; i++)
            markerManager.SpawnDebugMarker(markerParent, markers[i].WorldPosition, debugBoundsScale, debugBoundsColor, markers[i].Name);
    }

    QRTerrainMapper CreateTerrainMapper()
    {
        return new QRTerrainMapper(
            terrain,
            flipX,
            flipZ);
    }

    void PumpLiveTexture()
    {
        TryInitializeLiveFrameSource();

        if (frameTextureSource != null)
            frameTextureSource.PumpLatestFrame(filterMode, OnSourceTextureUpdated);
    }

    void PumpDebugTexture()
    {
        Texture2D debugTexture = QRDebugImageStore.SourceTexture;
        int debugTextureVersion = QRDebugImageStore.Version;

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
                Debug.LogError("QRDetectionRenderer: Source is not assigned.");
                missingSourceLogged = true;
            }
            return;
        }

        frameTextureSource = new QRFrameTextureSource(
            Source,
            _stream,
            _format,
            _streamIndex,
            flipHorizontallyBeforeDetection,
            flipVerticallyBeforeDetection);
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

    void HandleTerrainHeightmapChanged()
    {
        detectionsDirty = true;
    }

    static string GetDetectionLabel(QRDetection detection)
    {
        if (!string.IsNullOrEmpty(detection.decoded))
            return detection.decoded;
        if (!string.IsNullOrEmpty(detection.data))
            return detection.data;

        return "QR";
    }

    static string GetDepthCentroidPercentageLabel(QRDetection detection)
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
}
