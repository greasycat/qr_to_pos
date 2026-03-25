using System;
using UnityEngine;

[Serializable]
public struct MarkerDetection
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
    public MarkerDetection[] detections;
    public int count;
    public float processing_time;
    public string error;
}

public struct MarkerDebugPlacement
{
    public string Name;
    public Vector3 WorldPosition;

    public MarkerDebugPlacement(string name, Vector3 worldPosition)
    {
        Name = name;
        WorldPosition = worldPosition;
    }
}
