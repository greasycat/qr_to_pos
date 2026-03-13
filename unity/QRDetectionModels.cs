using System;
using UnityEngine;

[Serializable]
public struct QRDetection
{
    public string data;
    public int[] bbox;
    public float confidence;
    public string decoded;
    public float[] depth_centroid;
    public float[] depth_centroid_pct;
}

[Serializable]
public struct DetectionResponse
{
    public QRDetection[] detections;
    public int count;
    public float processing_time;
    public string error;
}

[Serializable]
public struct DetectionRequest
{
    public string action;
    public string image;
    public bool flip_horizontal;
}

public struct QRDebugMarkerPlacement
{
    public string Name;
    public Vector3 WorldPosition;

    public QRDebugMarkerPlacement(string name, Vector3 worldPosition)
    {
        Name = name;
        WorldPosition = worldPosition;
    }
}
