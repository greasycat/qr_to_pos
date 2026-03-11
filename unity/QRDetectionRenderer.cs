using Intel.RealSense;
using NativeWebSocket;
using System;
using System.Collections.Generic;
using UnityEngine;

public class QRDetectionRenderer : MonoBehaviour
{
    public RsFrameProvider Source;

    public Stream _stream;
    public Format _format;
    public int _streamIndex;

    public FilterMode filterMode = FilterMode.Point;

    [Header("WebSocket")]
    public string serverUrl = "ws://localhost:8765";
    public float sendInterval = 0.3f;

    [Header("Marker Placement")]
    public Terrain terrain;
    public Transform markerParent;
    public Vector3 markerScale = new Vector3(0.08f, 0.08f, 0.08f);
    public Color markerColor = Color.green;
    public float markerVerticalOffset = 0.05f;

    [Header("Debug Bounds")]
    public bool showDebugBounds = true;
    public Color debugBoundsColor = Color.blue;
    public Vector3 debugBoundsScale = new Vector3(0.12f, 0.12f, 0.12f);
    public float debugBoundsY = 0.5f;

    [Header("Depth Crop Mapping")]
    public int depthFrameWidth = 513;
    public int depthFrameHeight = 482;
    public int cropTop = 110;
    public int cropBottom = 170;
    public int cropLeft = 165;
    public int cropRight = 65;
    public bool flipX;
    public bool flipZ = true;

    FrameQueue q;
    Predicate<Frame> matcher;
    WebSocket websocket;

    Texture2D sourceTexture;

    float lastSendTime;
    bool sending;
    bool detectionsDirty;
    bool missingTerrainLogged;

    readonly List<QRDetection> detections = new List<QRDetection>();
    readonly object detectionsLock = new object();
    readonly List<GameObject> spawnedMarkers = new List<GameObject>();
    readonly List<GameObject> debugMarkers = new List<GameObject>();

    [Serializable]
    struct QRDetection
    {
        public string data;
        public int[] bbox; // [x1, y1, x2, y2]
        public float confidence;
        public string decoded;
    }

    [Serializable]
    struct DetectionResponse
    {
        public QRDetection[] detections;
        public int count;
        public float processing_time;
        public string error;
    }

    static TextureFormat Convert(Format lrsFormat)
    {
        switch (lrsFormat)
        {
            case Format.Z16: return TextureFormat.R16;
            case Format.Disparity16: return TextureFormat.R16;
            case Format.Rgb8: return TextureFormat.RGB24;
            case Format.Rgba8: return TextureFormat.RGBA32;
            case Format.Bgra8: return TextureFormat.BGRA32;
            case Format.Y8: return TextureFormat.Alpha8;
            case Format.Y16: return TextureFormat.R16;
            case Format.Raw16: return TextureFormat.R16;
            case Format.Raw8: return TextureFormat.Alpha8;
            case Format.Disparity32: return TextureFormat.RFloat;
            default:
                throw new ArgumentException(string.Format("librealsense format: {0}, is not supported by Unity", lrsFormat));
        }
    }

    static int BPP(TextureFormat format)
    {
        switch (format)
        {
            case TextureFormat.ARGB32:
            case TextureFormat.BGRA32:
            case TextureFormat.RGBA32:
                return 32;
            case TextureFormat.RGB24:
                return 24;
            case TextureFormat.R16:
                return 16;
            case TextureFormat.R8:
            case TextureFormat.Alpha8:
                return 8;
            default:
                throw new ArgumentException("unsupported format {0}", format.ToString());
        }
    }

    async void Start()
    {
        if (terrain == null)
            terrain = Terrain.activeTerrain;

        if (markerParent == null)
            markerParent = transform;

        if (Source == null)
        {
            Debug.LogError("QRDetectionRenderer: Source is not assigned.");
            return;
        }

        Source.OnStart += OnStartStreaming;
        Source.OnStop += OnStopStreaming;

        websocket = new WebSocket(serverUrl);

        websocket.OnOpen += () => Debug.Log("QRDetectionRenderer: WebSocket connected");
        websocket.OnError += (e) => Debug.LogError("QRDetectionRenderer: WebSocket error: " + e);
        websocket.OnClose += (e) => Debug.Log("QRDetectionRenderer: WebSocket closed");

        websocket.OnMessage += (bytes) =>
        {
            var json = System.Text.Encoding.UTF8.GetString(bytes);
            var response = JsonUtility.FromJson<DetectionResponse>(json);

            if (!string.IsNullOrEmpty(response.error))
            {
                Debug.LogError("QRDetectionRenderer: Server error: " + response.error);
                sending = false;
                return;
            }

            lock (detectionsLock)
            {
                detections.Clear();
                if (response.detections != null)
                    detections.AddRange(response.detections);
            }

            detectionsDirty = true;
            sending = false;
        };

        await websocket.Connect();
    }

    void OnDestroy()
    {
        if (Source != null)
        {
            Source.OnStart -= OnStartStreaming;
            Source.OnStop -= OnStopStreaming;
            Source.OnNewSample -= OnNewSample;
        }

        if (q != null)
        {
            q.Dispose();
            q = null;
        }

        if (sourceTexture != null)
            Destroy(sourceTexture);

        ClearMarkers();
        ClearDebugMarkers();
    }

    void OnStopStreaming()
    {
        if (Source != null)
            Source.OnNewSample -= OnNewSample;

        if (q != null)
        {
            q.Dispose();
            q = null;
        }
    }

    public void OnStartStreaming(PipelineProfile activeProfile)
    {
        q = new FrameQueue(1);
        matcher = new Predicate<Frame>(Matches);
        Source.OnNewSample += OnNewSample;
    }

    bool Matches(Frame f)
    {
        using (var p = f.Profile)
            return p.Stream == _stream && p.Format == _format && (p.Index == _streamIndex || _streamIndex == -1);
    }

    void OnNewSample(Frame frame)
    {
        try
        {
            if (frame.IsComposite)
            {
                using (var fs = frame.As<FrameSet>())
                using (var f = fs.FirstOrDefault(matcher))
                {
                    if (f != null)
                        q.Enqueue(f);

                    return;
                }
            }

            if (!matcher(frame))
                return;

            using (frame)
                q.Enqueue(frame);
        }
        catch (Exception e)
        {
            Debug.LogException(e);
        }
    }

    bool HasTextureConflict(VideoFrame vf, Texture2D tex)
    {
        return !tex ||
            tex.width != vf.Width ||
            tex.height != vf.Height ||
            BPP(tex.format) != vf.BitsPerPixel;
    }

    void LateUpdate()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        if (websocket != null)
            websocket.DispatchMessageQueue();
#endif

        if (q == null)
            return;

        VideoFrame frame;
        if (q.PollForFrame<VideoFrame>(out frame))
        {
            using (frame)
                ProcessFrame(frame);
        }

        if (detectionsDirty)
            RefreshMarkers();
    }

    void ProcessFrame(VideoFrame frame)
    {
        if (HasTextureConflict(frame, sourceTexture))
        {
            if (sourceTexture != null)
                Destroy(sourceTexture);

            using (var p = frame.Profile)
            {
                bool linear = (QualitySettings.activeColorSpace != ColorSpace.Linear)
                    || (p.Stream != Stream.Color && p.Stream != Stream.Infrared);
                sourceTexture = new Texture2D(frame.Width, frame.Height, Convert(p.Format), false, linear)
                {
                    wrapMode = TextureWrapMode.Clamp,
                    filterMode = filterMode
                };
            }
        }

        sourceTexture.LoadRawTextureData(frame.Data, frame.Stride * frame.Height);
        sourceTexture.Apply();

        SendFrameToServer();
    }

    async void SendFrameToServer()
    {
        if (sending)
            return;
        if (websocket == null || websocket.State != WebSocketState.Open)
            return;
        if (Time.time - lastSendTime < sendInterval)
            return;
        if (sourceTexture == null)
            return;

        sending = true;
        lastSendTime = Time.time;

        byte[] png = sourceTexture.EncodeToPNG();
        await websocket.Send(png);
    }

    void RefreshMarkers()
    {
        detectionsDirty = false;
        ClearMarkers();
        ClearDebugMarkers();

        if (terrain == null)
        {
            if (!missingTerrainLogged)
            {
                Debug.LogWarning("QRDetectionRenderer: No terrain assigned, skipping marker placement.");
                missingTerrainLogged = true;
            }
            return;
        }

        missingTerrainLogged = false;

        if (showDebugBounds)
            SpawnDebugBoundsMarkers();

        List<QRDetection> currentDetections;
        lock (detectionsLock)
        {
            currentDetections = new List<QRDetection>(detections);
        }

        for (int i = 0; i < currentDetections.Count; i++)
        {
            Vector3 worldPosition;
            if (!TryGetMarkerPosition(currentDetections[i], out worldPosition))
                continue;

            SpawnMarker(worldPosition, currentDetections[i], i);
        }
    }

    bool TryGetMarkerPosition(QRDetection detection, out Vector3 worldPosition)
    {
        worldPosition = Vector3.zero;

        if (terrain == null || sourceTexture == null)
            return false;
        if (detection.bbox == null || detection.bbox.Length < 4)
            return false;

        float centroidX = (detection.bbox[0] + detection.bbox[2]) * 0.5f;
        float centroidY = (detection.bbox[1] + detection.bbox[3]) * 0.5f;

        float xNorm;
        float zNorm;
        if (!TryGetTerrainNormalizedPosition(centroidX, centroidY, out xNorm, out zNorm))
            return false;

        TerrainData terrainData = terrain.terrainData;
        Vector3 terrainSize = terrainData.size;
        Vector3 terrainPos = terrain.GetPosition();

        float worldX = terrainPos.x + xNorm * terrainSize.x;
        float worldZ = terrainPos.z + zNorm * terrainSize.z;
        float worldY = terrain.SampleHeight(new Vector3(worldX, 0f, worldZ)) + terrainPos.y + markerVerticalOffset;

        worldPosition = new Vector3(worldX, worldY, worldZ);
        return true;
    }

    bool TryGetTerrainNormalizedPosition(float imageX, float imageY, out float xNorm, out float zNorm)
    {
        xNorm = 0f;
        zNorm = 0f;

        if (sourceTexture == null)
            return false;

        float frameWidth = sourceTexture.width;
        float frameHeight = sourceTexture.height;
        float scaleX = frameWidth / Mathf.Max(1, depthFrameWidth);
        float scaleY = frameHeight / Mathf.Max(1, depthFrameHeight);

        float depthColumn = imageX / scaleX;
        float depthRow = imageY / scaleY;

        float validRowCount = depthFrameHeight - cropTop - cropBottom;
        float validColumnCount = depthFrameWidth - cropLeft - cropRight;

        // TerrainEditor writes mesh[col - cropLeft + 1, row - cropTop], so the cropped
        // source occupies a transposed buffer with a one-column offset before resampling.
        float localRow = depthRow - cropTop;
        float localColumn = depthColumn - cropLeft + 1f;

        if (localRow < 0f || localRow > validRowCount - 1f)
            return false;
        if (localColumn < 1f || localColumn > validColumnCount)
            return false;

        // Image rows map to terrain X, image columns map to terrain Z.
        xNorm = localRow / Mathf.Max(1f, validRowCount);
        zNorm = localColumn / Mathf.Max(1f, validColumnCount);

        if (flipX)
            xNorm = 1f - xNorm;
        if (flipZ)
            zNorm = 1f - zNorm;
        return true;
    }

    void SpawnDebugBoundsMarkers()
    {
        float validRowCount = depthFrameHeight - cropTop - cropBottom;
        float validColumnCount = depthFrameWidth - cropLeft - cropRight;

        float minXNorm = 0f;
        float maxXNorm = (validRowCount - 1f) / Mathf.Max(1f, validRowCount);
        float minZNorm = 1f / Mathf.Max(1f, validColumnCount);
        float maxZNorm = 1f;

        SpawnDebugMarker(minXNorm, minZNorm, "QRBounds_MinMin");
        SpawnDebugMarker(minXNorm, maxZNorm, "QRBounds_MinMax");
        SpawnDebugMarker(maxXNorm, minZNorm, "QRBounds_MaxMin");
        SpawnDebugMarker(maxXNorm, maxZNorm, "QRBounds_MaxMax");
    }

    void SpawnDebugMarker(float xNorm, float zNorm, string markerName)
    {
        if (flipX)
            xNorm = 1f - xNorm;
        if (flipZ)
            zNorm = 1f - zNorm;

        TerrainData terrainData = terrain.terrainData;
        Vector3 terrainSize = terrainData.size;
        Vector3 terrainPos = terrain.GetPosition();

        float worldX = terrainPos.x + xNorm * terrainSize.x;
        float worldZ = terrainPos.z + zNorm * terrainSize.z;
        Vector3 worldPosition = new Vector3(worldX, debugBoundsY, worldZ);

        GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Cube);
        marker.name = markerName;
        marker.transform.SetParent(markerParent, true);
        marker.transform.position = worldPosition;
        marker.transform.localScale = debugBoundsScale;

        var markerRenderer = marker.GetComponent<Renderer>();
        if (markerRenderer != null)
            markerRenderer.material.color = debugBoundsColor;

        var markerCollider = marker.GetComponent<Collider>();
        if (markerCollider != null)
            markerCollider.enabled = false;

        debugMarkers.Add(marker);
    }

    void SpawnMarker(Vector3 position, QRDetection detection, int index)
    {
        GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Cube);
        marker.name = BuildMarkerName(detection, index);
        marker.transform.SetParent(markerParent, true);
        marker.transform.position = position;
        marker.transform.localScale = markerScale;

        var markerRenderer = marker.GetComponent<Renderer>();
        if (markerRenderer != null)
            markerRenderer.material.color = markerColor;

        var markerCollider = marker.GetComponent<Collider>();
        if (markerCollider != null)
            markerCollider.enabled = false;

        spawnedMarkers.Add(marker);
    }

    string BuildMarkerName(QRDetection detection, int index)
    {
        string label = detection.decoded;
        if (string.IsNullOrEmpty(label))
            label = detection.data;
        if (string.IsNullOrEmpty(label))
            label = "QR";

        return string.Format("QRMarker_{0}_{1}", index, label);
    }

    void ClearMarkers()
    {
        for (int i = 0; i < spawnedMarkers.Count; i++)
        {
            if (spawnedMarkers[i] != null)
                Destroy(spawnedMarkers[i]);
        }

        spawnedMarkers.Clear();
    }

    void ClearDebugMarkers()
    {
        for (int i = 0; i < debugMarkers.Count; i++)
        {
            if (debugMarkers[i] != null)
                Destroy(debugMarkers[i]);
        }

        debugMarkers.Clear();
    }

    async void OnApplicationQuit()
    {
        if (websocket != null && websocket.State == WebSocketState.Open)
            await websocket.Close();
    }
}
