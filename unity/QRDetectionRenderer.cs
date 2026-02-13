using Intel.RealSense;
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using NativeWebSocket;

public class QRDetectionRenderer : MonoBehaviour
{
    public RsFrameProvider Source;

    [System.Serializable]
    public class TextureEvent : UnityEvent<Texture> { }

    public Stream _stream;
    public Format _format;
    public int _streamIndex;

    public FilterMode filterMode = FilterMode.Point;

    [Header("WebSocket")]
    public string serverUrl = "ws://localhost:8765";
    public float sendInterval = 0.3f;

    [Header("Overlay")]
    public Color bboxColor = Color.green;
    public int bboxThickness = 2;
    public int fontSize = 20;

    [Space]
    public TextureEvent textureBinding;

    FrameQueue q;
    Predicate<Frame> matcher;
    WebSocket websocket;

    Texture2D sourceTexture;
    Texture2D overlayTexture;

    float lastSendTime;
    bool sending;

    List<QRDetection> detections = new List<QRDetection>();
    readonly object detectionsLock = new object();

    [Serializable]
    struct QRDetection
    {
        public string data;
        public int[] bbox; // [x, y, w, h]
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
            sending = false;
        };

        await websocket.Connect();
    }

    void OnDestroy()
    {
        if (sourceTexture != null) Destroy(sourceTexture);
        if (overlayTexture != null) Destroy(overlayTexture);
    }

    void OnStopStreaming()
    {
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
                    if (f != null) q.Enqueue(f);
                    return;
                }
            }
            if (!matcher(frame)) return;
            using (frame) q.Enqueue(frame);
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
        if (websocket != null) websocket.DispatchMessageQueue();
        #endif

        if (q == null) return;

        VideoFrame frame;
        if (q.PollForFrame<VideoFrame>(out frame))
            using (frame)
                ProcessFrame(frame);
    }

    void ProcessFrame(VideoFrame frame)
    {
        if (HasTextureConflict(frame, sourceTexture))
        {
            if (sourceTexture != null) Destroy(sourceTexture);
            if (overlayTexture != null) Destroy(overlayTexture);

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

            overlayTexture = new Texture2D(frame.Width, frame.Height, TextureFormat.RGBA32, false)
            {
                wrapMode = TextureWrapMode.Clamp,
                filterMode = filterMode
            };

            textureBinding.Invoke(overlayTexture);
        }

        sourceTexture.LoadRawTextureData(frame.Data, frame.Stride * frame.Height);
        sourceTexture.Apply();

        SendFrameToServer();
        RenderOverlay();
    }

    async void SendFrameToServer()
    {
        if (sending) return;
        if (websocket == null || websocket.State != WebSocketState.Open) return;
        if (Time.time - lastSendTime < sendInterval) return;

        sending = true;
        lastSendTime = Time.time;

        byte[] png = sourceTexture.EncodeToPNG();
        await websocket.Send(png);
    }

    void RenderOverlay()
    {
        // Copy source pixels into overlay, converting format
        Color32[] srcPixels = sourceTexture.GetPixels32();
        overlayTexture.SetPixels32(srcPixels);

        List<QRDetection> currentDetections;
        lock (detectionsLock)
        {
            currentDetections = new List<QRDetection>(detections);
        }

        if (currentDetections.Count == 0)
        {
            overlayTexture.Apply();
            return;
        }

        Color32[] pixels = overlayTexture.GetPixels32();
        int w = overlayTexture.width;
        int h = overlayTexture.height;
        Color32 color32 = bboxColor;

        foreach (var det in currentDetections)
        {
            if (det.bbox == null || det.bbox.Length < 4) continue;

            // bbox is [x1, y1, x2, y2] (corner coordinates)
            int x1 = det.bbox[0];
            int y1 = det.bbox[1];
            int x2 = det.bbox[2];
            int y2 = det.bbox[3];
            int bw = x2 - x1;
            int bh = y2 - y1;

            // Draw bounding box rectangle (texture origin is bottom-left, bbox origin is top-left)
            DrawRect(pixels, w, h, x1, y1, bw, bh, color32, bboxThickness);

            // Draw decoded text label above the bbox
            string label = det.decoded ?? det.data ?? "";
            if (!string.IsNullOrEmpty(label))
                DrawLabel(pixels, w, h, x1, y1 - fontSize - 4, label, color32);
        }

        overlayTexture.SetPixels32(pixels);
        overlayTexture.Apply();
    }

    void DrawRect(Color32[] pixels, int texW, int texH, int x, int y, int w, int h, Color32 color, int thickness)
    {
        for (int t = 0; t < thickness; t++)
        {
            // Top edge
            DrawHLine(pixels, texW, texH, x, y + t, w, color);
            // Bottom edge
            DrawHLine(pixels, texW, texH, x, y + h - 1 - t, w, color);
            // Left edge
            DrawVLine(pixels, texW, texH, x + t, y, h, color);
            // Right edge
            DrawVLine(pixels, texW, texH, x + w - 1 - t, y, h, color);
        }
    }

    void DrawHLine(Color32[] pixels, int texW, int texH, int x, int y, int length, Color32 color)
    {
        int flippedY = texH - 1 - y;
        for (int i = 0; i < length; i++)
        {
            int px = x + i;
            if (px < 0 || px >= texW || flippedY < 0 || flippedY >= texH) continue;
            pixels[flippedY * texW + px] = color;
        }
    }

    void DrawVLine(Color32[] pixels, int texW, int texH, int x, int y, int length, Color32 color)
    {
        for (int i = 0; i < length; i++)
        {
            int py = y + i;
            int flippedY = texH - 1 - py;
            if (x < 0 || x >= texW || flippedY < 0 || flippedY >= texH) continue;
            pixels[flippedY * texW + x] = color;
        }
    }

    void DrawLabel(Color32[] pixels, int texW, int texH, int x, int y, string text, Color32 color)
    {
        // Draw a background bar behind the text
        int charWidth = fontSize / 2;
        int labelW = text.Length * charWidth + 4;
        int labelH = fontSize + 4;

        Color32 bg = new Color32(0, 0, 0, 180);
        for (int dy = 0; dy < labelH; dy++)
        {
            for (int dx = 0; dx < labelW; dx++)
            {
                int px = x + dx;
                int py = y + dy;
                int flippedY = texH - 1 - py;
                if (px < 0 || px >= texW || flippedY < 0 || flippedY >= texH) continue;
                pixels[flippedY * texW + px] = bg;
            }
        }

        // Draw each character as a simple block letter
        for (int ci = 0; ci < text.Length; ci++)
        {
            DrawChar(pixels, texW, texH, x + 2 + ci * charWidth, y + 2, text[ci], fontSize, color);
        }
    }

    void DrawChar(Color32[] pixels, int texW, int texH, int x, int y, char c, int size, Color32 color)
    {
        ulong glyph = GetGlyph(c);
        // 5x7 grid scaled to size
        float scaleX = size / 2f / 5f;
        float scaleY = (float)size / 7f;

        for (int row = 0; row < 7; row++)
        {
            for (int col = 0; col < 5; col++)
            {
                int bit = row * 5 + col;
                if (((glyph >> bit) & 1) == 0) continue;

                int startX = x + (int)(col * scaleX);
                int endX = x + (int)((col + 1) * scaleX);
                int startY = y + (int)(row * scaleY);
                int endY = y + (int)((row + 1) * scaleY);

                for (int py = startY; py < endY; py++)
                {
                    int flippedY = texH - 1 - py;
                    if (flippedY < 0 || flippedY >= texH) continue;
                    for (int px = startX; px < endX; px++)
                    {
                        if (px < 0 || px >= texW) continue;
                        pixels[flippedY * texW + px] = color;
                    }
                }
            }
        }
    }

    static ulong GetGlyph(char c)
    {
        // 5x7 bitmap font stored as 35-bit values (LSB = top-left)
        switch (char.ToUpper(c))
        {
            case '0': return 0b01110_10001_10011_10101_11001_10001_01110UL;
            case '1': return 0b00100_01100_00100_00100_00100_00100_01110UL;
            case '2': return 0b01110_10001_00001_00110_01000_10000_11111UL;
            case '3': return 0b01110_10001_00001_00110_00001_10001_01110UL;
            case '4': return 0b00010_00110_01010_10010_11111_00010_00010UL;
            case '5': return 0b11111_10000_11110_00001_00001_10001_01110UL;
            case '6': return 0b00110_01000_10000_11110_10001_10001_01110UL;
            case '7': return 0b11111_00001_00010_00100_01000_01000_01000UL;
            case '8': return 0b01110_10001_10001_01110_10001_10001_01110UL;
            case '9': return 0b01110_10001_10001_01111_00001_00010_01100UL;
            case 'A': return 0b01110_10001_10001_11111_10001_10001_10001UL;
            case 'B': return 0b11110_10001_10001_11110_10001_10001_11110UL;
            case 'C': return 0b01110_10001_10000_10000_10000_10001_01110UL;
            case 'D': return 0b11110_10001_10001_10001_10001_10001_11110UL;
            case 'E': return 0b11111_10000_10000_11110_10000_10000_11111UL;
            case 'F': return 0b11111_10000_10000_11110_10000_10000_10000UL;
            case 'G': return 0b01110_10001_10000_10111_10001_10001_01110UL;
            case 'H': return 0b10001_10001_10001_11111_10001_10001_10001UL;
            case 'I': return 0b01110_00100_00100_00100_00100_00100_01110UL;
            case 'J': return 0b00111_00010_00010_00010_10010_10010_01100UL;
            case 'K': return 0b10001_10010_10100_11000_10100_10010_10001UL;
            case 'L': return 0b10000_10000_10000_10000_10000_10000_11111UL;
            case 'M': return 0b10001_11011_10101_10101_10001_10001_10001UL;
            case 'N': return 0b10001_11001_10101_10011_10001_10001_10001UL;
            case 'O': return 0b01110_10001_10001_10001_10001_10001_01110UL;
            case 'P': return 0b11110_10001_10001_11110_10000_10000_10000UL;
            case 'Q': return 0b01110_10001_10001_10001_10101_10010_01101UL;
            case 'R': return 0b11110_10001_10001_11110_10100_10010_10001UL;
            case 'S': return 0b01110_10001_10000_01110_00001_10001_01110UL;
            case 'T': return 0b11111_00100_00100_00100_00100_00100_00100UL;
            case 'U': return 0b10001_10001_10001_10001_10001_10001_01110UL;
            case 'V': return 0b10001_10001_10001_10001_01010_01010_00100UL;
            case 'W': return 0b10001_10001_10001_10101_10101_10101_01010UL;
            case 'X': return 0b10001_10001_01010_00100_01010_10001_10001UL;
            case 'Y': return 0b10001_10001_01010_00100_00100_00100_00100UL;
            case 'Z': return 0b11111_00001_00010_00100_01000_10000_11111UL;
            case '/': return 0b00001_00010_00010_00100_01000_01000_10000UL;
            case ':': return 0b00000_00100_00100_00000_00100_00100_00000UL;
            case '.': return 0b00000_00000_00000_00000_00000_01100_01100UL;
            case '-': return 0b00000_00000_00000_11111_00000_00000_00000UL;
            case '_': return 0b00000_00000_00000_00000_00000_00000_11111UL;
            case '?': return 0b01110_10001_00001_00110_00100_00000_00100UL;
            case ' ': return 0UL;
            default:  return 0b10101_01010_10101_01010_10101_01010_10101UL; // checkerboard for unknown
        }
    }

    async void OnApplicationQuit()
    {
        if (websocket != null && websocket.State == WebSocketState.Open)
            await websocket.Close();
    }
}
