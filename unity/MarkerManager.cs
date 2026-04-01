using System.Collections.Generic;
using UnityEngine;

public sealed class MarkerManager
{
    readonly Dictionary<string, TrackedMarker> trackedMarkers = new Dictionary<string, TrackedMarker>();
    readonly Dictionary<string, GameObject> debugMarkers = new Dictionary<string, GameObject>();
    readonly HashSet<string> activeDebugMarkers = new HashSet<string>();

    public void TrackMarker(
        Transform markerParent,
        Vector3 targetPosition,
        Vector3 markerScale,
        MarkerDetection detection,
        bool enableFall,
        float fallSpawnHeight,
        float fallAcceleration,
        float maxFallSpeed)
    {
        PruneMissingMarkers();

        string tagKey = GetTrackingKey(detection);
        string markerName = BuildMarkerName(detection, tagKey);
        Color markerColor = GetMarkerColor(detection, tagKey);

        TrackedMarker trackedMarker;
        GameObject marker;
        bool isNewMarker = !trackedMarkers.TryGetValue(tagKey, out trackedMarker) || trackedMarker.Marker == null;
        if (isNewMarker)
        {
            marker = CreateMarker(
                markerParent,
                markerName,
                targetPosition,
                markerScale,
                markerColor);
            trackedMarker = new TrackedMarker(marker);
            trackedMarkers[tagKey] = trackedMarker;
        }
        else
        {
            marker = trackedMarker.Marker;
        }

        UpdateMarker(marker, markerParent, markerName, markerScale, markerColor);
        UpdateFallController(
            marker,
            targetPosition,
            enableFall,
            fallSpawnHeight,
            fallAcceleration,
            maxFallSpeed);
        trackedMarker.LastSeenTime = Time.time;
    }

    public void BeginDebugMarkerRefresh()
    {
        activeDebugMarkers.Clear();
    }

    public void SpawnDebugMarker(Transform markerParent, Vector3 position, Vector3 debugBoundsScale, Color debugBoundsColor, string markerName)
    {
        activeDebugMarkers.Add(markerName);

        GameObject marker;
        if (!debugMarkers.TryGetValue(markerName, out marker) || marker == null)
        {
            marker = CreateMarker(markerParent, markerName, position, debugBoundsScale, debugBoundsColor);
            debugMarkers[markerName] = marker;
            return;
        }

        marker.transform.position = position;
        UpdateMarker(marker, markerParent, markerName, debugBoundsScale, debugBoundsColor);
    }

    public void EndDebugMarkerRefresh()
    {
        if (debugMarkers.Count == 0)
            return;

        var staleMarkerNames = new List<string>();
        foreach (var entry in debugMarkers)
        {
            if (!activeDebugMarkers.Contains(entry.Key))
            {
                if (entry.Value != null)
                    Object.Destroy(entry.Value);

                staleMarkerNames.Add(entry.Key);
            }
        }

        for (int i = 0; i < staleMarkerNames.Count; i++)
            debugMarkers.Remove(staleMarkerNames[i]);
    }

    public void ClearMarkers()
    {
        foreach (var entry in trackedMarkers)
        {
            if (entry.Value != null && entry.Value.Marker != null)
                Object.Destroy(entry.Value.Marker);
        }

        trackedMarkers.Clear();
    }

    public void ClearDebugMarkers()
    {
        foreach (var entry in debugMarkers)
        {
            if (entry.Value != null)
                Object.Destroy(entry.Value);
        }

        debugMarkers.Clear();
        activeDebugMarkers.Clear();
    }

    public void ClearAll()
    {
        ClearMarkers();
        ClearDebugMarkers();
    }

    public void PruneExpiredMarkers(float currentTime, float markerLifetimeSeconds)
    {
        if (trackedMarkers.Count == 0)
            return;

        if (markerLifetimeSeconds <= 0f || float.IsInfinity(markerLifetimeSeconds))
            return;

        float lifetime = Mathf.Max(0f, markerLifetimeSeconds);
        var expiredKeys = new List<string>();
        foreach (var entry in trackedMarkers)
        {
            TrackedMarker trackedMarker = entry.Value;
            if (trackedMarker == null || trackedMarker.Marker == null)
            {
                expiredKeys.Add(entry.Key);
                continue;
            }

            if (currentTime - trackedMarker.LastSeenTime > lifetime)
            {
                Object.Destroy(trackedMarker.Marker);
                expiredKeys.Add(entry.Key);
            }
        }

        for (int i = 0; i < expiredKeys.Count; i++)
            trackedMarkers.Remove(expiredKeys[i]);
    }

    void PruneMissingMarkers()
    {
        if (trackedMarkers.Count == 0)
            return;

        var missingKeys = new List<string>();
        foreach (var entry in trackedMarkers)
        {
            if (entry.Value == null || entry.Value.Marker == null)
                missingKeys.Add(entry.Key);
        }

        for (int i = 0; i < missingKeys.Count; i++)
            trackedMarkers.Remove(missingKeys[i]);
    }

    GameObject CreateMarker(Transform markerParent, string markerName, Vector3 position, Vector3 scale, Color color)
    {
        GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Cube);
        marker.transform.position = position;
        UpdateMarker(marker, markerParent, markerName, scale, color);
        return marker;
    }

    void UpdateMarker(GameObject marker, Transform markerParent, string markerName, Vector3 scale, Color color)
    {
        marker.name = markerName;
        marker.transform.SetParent(markerParent, true);
        marker.transform.localScale = scale;

        var markerRenderer = marker.GetComponent<Renderer>();
        if (markerRenderer != null)
            markerRenderer.material.color = color;
    }

    public static string GetTrackingKey(MarkerDetection detection)
    {
        if (!string.IsNullOrEmpty(detection.data))
            return detection.data;
        if (!string.IsNullOrEmpty(detection.decoded))
            return detection.decoded;

        return "Marker";
    }

    static Color GetMarkerColor(MarkerDetection detection, string tagKey)
    {
        int tagId;
        if (int.TryParse(detection.data, out tagId))
            return BuildMarkerColorFromSeed(unchecked((uint)tagId));

        return BuildMarkerColorFromSeed(ComputeStableHash(tagKey));
    }

    static Color BuildMarkerColorFromSeed(uint seed)
    {
        uint mixedSeed = seed * 2654435761u + 2246822519u;
        float hue = (mixedSeed % 360u) / 360f;
        float saturation = 0.65f + ((mixedSeed >> 9) % 20u) / 100f;
        float value = 0.85f + ((mixedSeed >> 17) % 10u) / 100f;
        return Color.HSVToRGB(hue, saturation, value);
    }

    static uint ComputeStableHash(string value)
    {
        const uint offsetBasis = 2166136261u;
        const uint prime = 16777619u;

        if (string.IsNullOrEmpty(value))
            return offsetBasis;

        uint hash = offsetBasis;
        for (int i = 0; i < value.Length; i++)
        {
            hash ^= value[i];
            hash *= prime;
        }
        return hash;
    }

    static string BuildMarkerName(MarkerDetection detection, string tagKey)
    {
        string label = detection.decoded;
        if (string.IsNullOrEmpty(label))
            label = tagKey;

        return string.Format("Marker_{0}", label);
    }

    static void UpdateFallController(
        GameObject marker,
        Vector3 targetPosition,
        bool enableFall,
        float fallSpawnHeight,
        float fallAcceleration,
        float maxFallSpeed)
    {
        var fallController = marker.GetComponent<MarkerFallController>();
        if (!enableFall)
        {
            if (fallController != null)
                Object.Destroy(fallController);

            var markerRigidbody = marker.GetComponent<Rigidbody>();
            if (markerRigidbody != null)
                Object.Destroy(markerRigidbody);

            marker.transform.position = targetPosition;
            marker.transform.rotation = Quaternion.identity;
            return;
        }

        if (fallController == null)
            fallController = marker.AddComponent<MarkerFallController>();

        fallController.Configure(fallSpawnHeight, fallAcceleration, maxFallSpeed);
        fallController.SetTarget(targetPosition, false);
    }

    sealed class TrackedMarker
    {
        public readonly GameObject Marker;
        public float LastSeenTime;

        public TrackedMarker(GameObject marker)
        {
            Marker = marker;
            LastSeenTime = Time.time;
        }
    }
}
