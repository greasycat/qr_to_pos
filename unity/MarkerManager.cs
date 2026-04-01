using System.Collections.Generic;
using UnityEngine;

public sealed class MarkerManager
{
    const float MarkerLifetimeSeconds = 3f;
    const float MaxMarkerMovementDistancePerUpdate = 5f;
    const float PlacementRaycastPadding = 0.01f;

    readonly Dictionary<string, TrackedMarker> trackedMarkers = new Dictionary<string, TrackedMarker>();
    readonly Dictionary<string, GameObject> debugMarkers = new Dictionary<string, GameObject>();
    readonly HashSet<string> activeDebugMarkers = new HashSet<string>();

    public void TrackMarker(
        Transform markerParent,
        Vector3 targetPosition,
        Vector3 markerScale,
        MarkerDetection detection,
        GameObject markerPrefab,
        bool removeColorWhenUsingPrefab)
    {
        string tagKey;
        if (!TryGetTrackingKey(detection, out tagKey))
            return;

        string markerName = BuildMarkerName(detection, tagKey);
        Color markerColor = GetMarkerColor(detection, tagKey);

        TrackedMarker trackedMarker;
        GameObject marker;
        bool isNewMarker = !trackedMarkers.TryGetValue(tagKey, out trackedMarker)
            || trackedMarker == null
            || trackedMarker.Marker == null;
        if (isNewMarker)
        {
            marker = CreateMarker(
                markerParent,
                markerName,
                targetPosition,
                markerScale,
                markerColor,
                markerPrefab);
            trackedMarker = new TrackedMarker(marker, markerPrefab != null);
            trackedMarkers[tagKey] = trackedMarker;
        }
        else
        {
            marker = trackedMarker.Marker;
        }

        bool applyColor = !trackedMarker.UsesPrefab || !removeColorWhenUsingPrefab;
        UpdateMarker(marker, markerParent, markerName, markerScale, markerColor, applyColor);
        marker.transform.rotation = Quaternion.identity;
        Vector3 nextTargetPosition = isNewMarker
            ? targetPosition
            : Vector3.MoveTowards(marker.transform.position, targetPosition, MaxMarkerMovementDistancePerUpdate);
        marker.transform.position = nextTargetPosition;
        marker.transform.position = ResolvePlacementPosition(marker, nextTargetPosition);
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
            marker = CreateMarker(markerParent, markerName, position, debugBoundsScale, debugBoundsColor, null);
            debugMarkers[markerName] = marker;
            return;
        }

        marker.transform.position = position;
        UpdateMarker(marker, markerParent, markerName, debugBoundsScale, debugBoundsColor, true);
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

    public void PruneExpiredMarkers(float currentTime)
    {
        if (trackedMarkers.Count == 0)
            return;

        var expiredKeys = new List<string>();
        foreach (var entry in trackedMarkers)
        {
            TrackedMarker trackedMarker = entry.Value;
            if (trackedMarker == null || trackedMarker.Marker == null)
            {
                expiredKeys.Add(entry.Key);
                continue;
            }

            if (currentTime - trackedMarker.LastSeenTime > MarkerLifetimeSeconds)
            {
                Object.Destroy(trackedMarker.Marker);
                expiredKeys.Add(entry.Key);
            }
        }

        for (int i = 0; i < expiredKeys.Count; i++)
            trackedMarkers.Remove(expiredKeys[i]);
    }

    GameObject CreateMarker(Transform markerParent, string markerName, Vector3 position, Vector3 scale, Color color, GameObject markerPrefab)
    {
        GameObject marker = markerPrefab != null
            ? Object.Instantiate(markerPrefab, position, Quaternion.identity)
            : GameObject.CreatePrimitive(PrimitiveType.Cube);
        marker.transform.position = position;
        UpdateMarker(marker, markerParent, markerName, scale, color, true);
        return marker;
    }

    void UpdateMarker(GameObject marker, Transform markerParent, string markerName, Vector3 scale, Color color, bool applyColor)
    {
        marker.name = markerName;
        marker.transform.SetParent(markerParent, true);
        marker.transform.localScale = scale;

        if (!applyColor)
            return;

        Renderer[] markerRenderers = marker.GetComponentsInChildren<Renderer>(true);
        for (int i = 0; i < markerRenderers.Length; i++)
            markerRenderers[i].material.color = color;
    }

    public static bool TryGetTrackingKey(MarkerDetection detection, out string tagKey)
    {
        if (!string.IsNullOrEmpty(detection.data))
        {
            tagKey = detection.data;
            return true;
        }

        tagKey = string.Empty;
        return false;
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

    static Vector3 ResolvePlacementPosition(GameObject marker, Vector3 startPosition)
    {
        Bounds markerBounds;
        if (!TryGetMarkerBounds(marker, out markerBounds))
            return startPosition;

        float halfHeight = markerBounds.extents.y;
        Vector3 rayOrigin = startPosition + Vector3.up * (halfHeight + PlacementRaycastPadding);
        RaycastHit[] hits = Physics.RaycastAll(
            rayOrigin,
            Vector3.down,
            Mathf.Infinity,
            ~0,
            QueryTriggerInteraction.Ignore);

        bool hasHit = false;
        float nearestDistance = float.PositiveInfinity;
        float surfaceY = startPosition.y;
        for (int i = 0; i < hits.Length; i++)
        {
            if (hits[i].collider == null || hits[i].collider.transform.IsChildOf(marker.transform))
                continue;

            if (hits[i].distance < nearestDistance)
            {
                nearestDistance = hits[i].distance;
                surfaceY = hits[i].point.y;
                hasHit = true;
            }
        }

        if (!hasHit)
            return startPosition;

        return new Vector3(startPosition.x, surfaceY + halfHeight, startPosition.z);
    }

    static bool TryGetMarkerBounds(GameObject marker, out Bounds bounds)
    {
        Collider[] colliders = marker.GetComponentsInChildren<Collider>(true);
        if (TryEncapsulateBounds(colliders, out bounds))
            return true;

        Renderer[] renderers = marker.GetComponentsInChildren<Renderer>(true);
        return TryEncapsulateBounds(renderers, out bounds);
    }

    static bool TryEncapsulateBounds<T>(T[] components, out Bounds bounds) where T : Component
    {
        bounds = default;
        bool hasBounds = false;
        for (int i = 0; i < components.Length; i++)
        {
            if (components[i] == null)
                continue;

            Bounds componentBounds;
            if (components[i] is Collider collider)
                componentBounds = collider.bounds;
            else if (components[i] is Renderer renderer)
                componentBounds = renderer.bounds;
            else
                continue;

            if (!hasBounds)
            {
                bounds = componentBounds;
                hasBounds = true;
            }
            else
            {
                bounds.Encapsulate(componentBounds);
            }
        }

        return hasBounds;
    }

    sealed class TrackedMarker
    {
        public readonly GameObject Marker;
        public readonly bool UsesPrefab;
        public float LastSeenTime;

        public TrackedMarker(GameObject marker, bool usesPrefab)
        {
            Marker = marker;
            UsesPrefab = usesPrefab;
            LastSeenTime = Time.time;
        }
    }
}
