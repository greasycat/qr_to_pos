using UnityEngine;

public static class QRDebugImageStore
{
    static Texture2D sourceTexture;
    static int version;

    public static Texture2D SourceTexture
    {
        get { return sourceTexture; }
    }

    public static int Version
    {
        get { return version; }
    }

    public static string SourceLabel { get; private set; }

    public static bool TrySetImage(byte[] imageBytes, string sourceLabel, FilterMode filterMode = FilterMode.Point)
    {
        if (imageBytes == null || imageBytes.Length == 0)
            return false;

        if (sourceTexture == null)
        {
            sourceTexture = new Texture2D(2, 2, TextureFormat.RGBA32, false)
            {
                wrapMode = TextureWrapMode.Clamp
            };
        }

        if (!ImageConversion.LoadImage(sourceTexture, imageBytes, false))
            return false;

        sourceTexture.wrapMode = TextureWrapMode.Clamp;
        sourceTexture.filterMode = filterMode;
        SourceLabel = sourceLabel;
        version++;
        return true;
    }

    public static void Clear()
    {
        SourceLabel = null;
        version++;

        if (sourceTexture == null)
            return;

#if UNITY_EDITOR
        Object.DestroyImmediate(sourceTexture);
#else
        Object.Destroy(sourceTexture);
#endif
        sourceTexture = null;
    }
}
