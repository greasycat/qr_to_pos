using System.IO;
using UnityEditor;
using UnityEngine;

public sealed class MarkerManualImageLoaderWindow : EditorWindow
{
    const string Title = "Marker Debug Image Loader";

    Vector2 scrollPosition;

    [MenuItem("Window/Markers/Debug Image Loader")]
    static void OpenWindow()
    {
        GetWindow<MarkerManualImageLoaderWindow>(Title);
    }

    void OnGUI()
    {
        EditorGUILayout.LabelField("Manual Marker Image Loader", EditorStyles.boldLabel);
        EditorGUILayout.HelpBox("Load a still image when the camera feed is unavailable. Enable debug mode on MarkerDetectionRenderer to use this texture.", MessageType.Info);

        using (new EditorGUILayout.HorizontalScope())
        {
            if (GUILayout.Button("Load Image", GUILayout.Height(28f)))
                LoadImageFromDisk();

            if (GUILayout.Button("Clear", GUILayout.Height(28f)))
            {
                MarkerDebugImageStore.Clear();
                Repaint();
            }
        }

        Texture2D texture = MarkerDebugImageStore.SourceTexture;
        string sourceLabel = string.IsNullOrEmpty(MarkerDebugImageStore.SourceLabel) ? "No image loaded" : MarkerDebugImageStore.SourceLabel;
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Current Image", sourceLabel);

        if (texture == null)
            return;

        EditorGUILayout.LabelField(string.Format("{0} x {1}", texture.width, texture.height));

        scrollPosition = EditorGUILayout.BeginScrollView(scrollPosition);
        Rect previewRect = GUILayoutUtility.GetAspectRect((float)texture.width / texture.height, GUILayout.ExpandWidth(true));
        EditorGUI.DrawPreviewTexture(previewRect, texture, null, ScaleMode.ScaleToFit);
        EditorGUILayout.EndScrollView();
    }

    void LoadImageFromDisk()
    {
        string path = EditorUtility.OpenFilePanel("Load Marker Debug Image", Application.dataPath, "png,jpg,jpeg");
        if (string.IsNullOrEmpty(path))
            return;

        byte[] imageBytes;
        try
        {
            imageBytes = File.ReadAllBytes(path);
        }
        catch (IOException exception)
        {
            Debug.LogException(exception);
            return;
        }

        if (!MarkerDebugImageStore.TrySetImage(imageBytes, Path.GetFileName(path)))
        {
            Debug.LogError("MarkerManualImageLoaderWindow: Failed to load image.");
            return;
        }

        Repaint();
    }
}
