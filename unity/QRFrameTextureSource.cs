using Intel.RealSense;
using System;
using UnityEngine;

public sealed class QRFrameTextureSource : IDisposable
{
    readonly RsFrameProvider source;
    readonly Stream stream;
    readonly Format format;
    readonly int streamIndex;

    FrameQueue queue;
    Predicate<Frame> matcher;

    public Texture2D SourceTexture { get; private set; }

    public QRFrameTextureSource(RsFrameProvider source, Stream stream, Format format, int streamIndex)
    {
        this.source = source;
        this.stream = stream;
        this.format = format;
        this.streamIndex = streamIndex;
    }

    public void Initialize()
    {
        DisposeQueue();
        queue = new FrameQueue(1);
        matcher = new Predicate<Frame>(Matches);
        source.OnNewSample -= OnNewSample;
        source.OnNewSample += OnNewSample;
        source.OnStart += OnStartStreaming;
        source.OnStop += OnStopStreaming;
    }

    public void PumpLatestFrame(FilterMode filterMode, Action<Texture2D> onTextureUpdated)
    {
        if (queue == null){
            // Debug.Log("queue null");
            return;
        }

        VideoFrame frame;
        if (!queue.PollForFrame<VideoFrame>(out frame))
        {
            // Debug.Log("wait for polling info");
            return;
        }

        using (frame)
        {
            ProcessFrame(frame, filterMode);
            if (onTextureUpdated != null)
                onTextureUpdated(SourceTexture);
        }
    }

    public void Dispose()
    {
        source.OnStart -= OnStartStreaming;
        source.OnStop -= OnStopStreaming;
        source.OnNewSample -= OnNewSample;

        DisposeQueue();

        if (SourceTexture != null)
        {
            UnityEngine.Object.Destroy(SourceTexture);
            SourceTexture = null;
        }
    }

    void OnStartStreaming(PipelineProfile activeProfile)
    {
        Debug.Log("OnStartStreaming in QRFrame");
        DisposeQueue();
        queue = new FrameQueue(1);
        matcher = new Predicate<Frame>(Matches);
        source.OnNewSample -= OnNewSample;
        source.OnNewSample += OnNewSample;
    }

    void OnStopStreaming()
    {
        source.OnNewSample -= OnNewSample;
        DisposeQueue();
    }

    bool Matches(Frame frame)
    {
        using (var profile = frame.Profile)
            return profile.Stream == stream && profile.Format == format && (profile.Index == streamIndex || streamIndex == -1);
    }

    void OnNewSample(Frame frame)
    {
        try
        {
            if (frame.IsComposite)
            {
                using (var frames = frame.As<FrameSet>())
                using (var matchedFrame = frames.FirstOrDefault(matcher))
                {
                    if (matchedFrame != null)
                        queue.Enqueue(matchedFrame);

                    return;
                }
            }

            if (!matcher(frame))
                return;

            using (frame)
                queue.Enqueue(frame);
        }
        catch (Exception e)
        {
            Debug.LogException(e);
        }
    }

    void ProcessFrame(VideoFrame frame, FilterMode filterMode)
    {
        if (HasTextureConflict(frame))
            RecreateTexture(frame, filterMode);

        SourceTexture.LoadRawTextureData(frame.Data, frame.Stride * frame.Height);
        SourceTexture.Apply();
    }

    bool HasTextureConflict(VideoFrame frame)
    {
        return SourceTexture == null
            || SourceTexture.width != frame.Width
            || SourceTexture.height != frame.Height
            || BitsPerPixel(SourceTexture.format) != frame.BitsPerPixel;
    }

    void RecreateTexture(VideoFrame frame, FilterMode filterMode)
    {
        if (SourceTexture != null)
            UnityEngine.Object.Destroy(SourceTexture);

        using (var profile = frame.Profile)
        {
            bool linear = (QualitySettings.activeColorSpace != ColorSpace.Linear)
                || (profile.Stream != Stream.Color && profile.Stream != Stream.Infrared);
            SourceTexture = new Texture2D(frame.Width, frame.Height, ConvertFormat(profile.Format), false, linear)
            {
                wrapMode = TextureWrapMode.Clamp,
                filterMode = filterMode
            };
        }
    }

    void DisposeQueue()
    {
        if (queue == null)
            return;

        queue.Dispose();
        queue = null;
    }

    static TextureFormat ConvertFormat(Format lrsFormat)
    {
        switch (lrsFormat)
        {
            case Format.Z16:
            case Format.Disparity16:
            case Format.Y16:
            case Format.Raw16:
                return TextureFormat.R16;
            case Format.Rgb8:
                return TextureFormat.RGB24;
            case Format.Rgba8:
                return TextureFormat.RGBA32;
            case Format.Bgra8:
                return TextureFormat.BGRA32;
            case Format.Y8:
            case Format.Raw8:
                return TextureFormat.Alpha8;
            case Format.Disparity32:
                return TextureFormat.RFloat;
            default:
                throw new ArgumentException(string.Format("librealsense format: {0}, is not supported by Unity", lrsFormat));
        }
    }

    static int BitsPerPixel(TextureFormat textureFormat)
    {
        switch (textureFormat)
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
                throw new ArgumentException("unsupported format {0}", textureFormat.ToString());
        }
    }
}
