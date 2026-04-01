using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public sealed class MarkerManager
{
    const float DefaultRecentDetectionLifetimeSeconds = 5f;
    const float DefaultRecentDetectionDistance = 10f;

    readonly List<GameObject> spawnedMarkers = new List<GameObject>();
    readonly List<RecentDetectionCentroid> recentDetectionCentroids = new List<RecentDetectionCentroid>();
    readonly Dictionary<string, GameObject> debugMarkers = new Dictionary<string, GameObject>();
    readonly HashSet<string> activeDebugMarkers = new HashSet<string>();

    MarkerManagerCoroutineHost coroutineHost;

    public void SpawnMarker(
        Transform markerParent,
        Vector3 targetPosition,
        Vector3 markerScale,
        Color markerColor,
        MarkerDetection detection,
        int index,
        bool enableFall,
        float fallSpawnHeight,
        float fallAcceleration,
        float maxFallSpeed,
        float recentDetectionLifetimeSeconds = DefaultRecentDetectionLifetimeSeconds,
        float recentDetectionDistance = DefaultRecentDetectionDistance)
    {
        EnsureCoroutineHost(markerParent);
        PruneMissingMarkers();

        if (!CanSpawnAtWorldPosition(targetPosition, recentDetectionDistance))
            return;

        GameObject marker = CreateMarker(
            markerParent,
            BuildMarkerName(detection, index),
            GetSpawnPosition(targetPosition, fallSpawnHeight, enableFall),
            markerScale,
            markerColor);

        spawnedMarkers.Add(marker);
        RegisterRecentDetectionCentroid(targetPosition, recentDetectionLifetimeSeconds);
        UpdateFallController(marker, targetPosition, enableFall, fallSpawnHeight, fallAcceleration, maxFallSpeed);
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

        UpdateMarker(marker, markerParent, position, debugBoundsScale, debugBoundsColor);
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
        for (int i = 0; i < spawnedMarkers.Count; i++)
        {
            if (spawnedMarkers[i] != null)
                Object.Destroy(spawnedMarkers[i]);
        }

        if (coroutineHost != null)
            coroutineHost.StopAllCoroutines();

        spawnedMarkers.Clear();
        recentDetectionCentroids.Clear();
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

    bool CanSpawnAtWorldPosition(Vector3 targetPosition, float recentDetectionDistance)
    {
        float maxDistance = Mathf.Max(0f, recentDetectionDistance);
        float maxDistanceSqr = maxDistance * maxDistance;

        for (int i = spawnedMarkers.Count - 1; i >= 0; i--)
        {
            GameObject marker = spawnedMarkers[i];
            if (marker == null)
                continue;

            if (IsWithinSpawnDistance(marker.transform.position, targetPosition, maxDistanceSqr))
                return false;
        }

        float totalDistance = 0f;
        for (int i = recentDetectionCentroids.Count - 1; i >= 0; i--)
        {
            float d = (recentDetectionCentroids[i].WorldPosition - targetPosition).sqrMagnitude;
            totalDistance += d;
            if (IsWithinSpawnDistance(recentDetectionCentroids[i].WorldPosition, targetPosition, maxDistanceSqr))
                return false;
        }

        Debug.Log($"Spawn filter distances totalSqr={totalDistance}, activeMarkers={spawnedMarkers.Count}, recentCentroids={recentDetectionCentroids.Count}");

        return true;
    }

    void RegisterRecentDetectionCentroid(Vector3 worldPosition, float recentDetectionLifetimeSeconds)
    {
        var centroid = new RecentDetectionCentroid(worldPosition);
        recentDetectionCentroids.Add(centroid);

        if (coroutineHost == null)
            return;

        centroid.ExpiryCoroutine = coroutineHost.StartCoroutine(
            ExpireRecentDetectionCentroidAfterDelay(centroid, Mathf.Max(0f, recentDetectionLifetimeSeconds)));
    }

    IEnumerator ExpireRecentDetectionCentroidAfterDelay(RecentDetectionCentroid centroid, float delaySeconds)
    {
        if (delaySeconds > 0f)
            yield return new WaitForSeconds(delaySeconds);

        recentDetectionCentroids.Remove(centroid);
    }

    void EnsureCoroutineHost(Transform markerParent)
    {
        if (markerParent == null)
            return;

        if (coroutineHost != null)
            return;

        coroutineHost = markerParent.GetComponent<MarkerManagerCoroutineHost>();
        if (coroutineHost == null)
            coroutineHost = markerParent.gameObject.AddComponent<MarkerManagerCoroutineHost>();
    }

    void PruneMissingMarkers()
    {
        for (int i = spawnedMarkers.Count - 1; i >= 0; i--)
        {
            if (spawnedMarkers[i] == null)
                spawnedMarkers.RemoveAt(i);
        }
    }

    GameObject CreateMarker(Transform markerParent, string markerName, Vector3 position, Vector3 scale, Color color)
    {
        GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Cube);
        marker.name = markerName;
        UpdateMarker(marker, markerParent, position, scale, color);
        return marker;
    }

    void UpdateMarker(GameObject marker, Transform markerParent, Vector3 position, Vector3 scale, Color color)
    {
        marker.transform.SetParent(markerParent, true);
        marker.transform.position = position;
        marker.transform.localScale = scale;

        var markerRenderer = marker.GetComponent<Renderer>();
        if (markerRenderer != null)
            markerRenderer.material.color = color;
    }

    static string BuildMarkerName(MarkerDetection detection, int index)
    {
        string label = detection.decoded;
        if (string.IsNullOrEmpty(label))
            label = detection.data;
        if (string.IsNullOrEmpty(label))
            label = "Marker";

        return string.Format("Marker_{0}_{1}", index, label);
    }

    static Vector3 GetSpawnPosition(Vector3 targetPosition, float fallSpawnHeight, bool enableFall)
    {
        if (!enableFall)
            return targetPosition;

        return new Vector3(targetPosition.x, targetPosition.y + Mathf.Max(0f, fallSpawnHeight), targetPosition.z);
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
        fallController.SetTarget(targetPosition, true);
    }

    static bool IsWithinSpawnDistance(Vector3 a, Vector3 b, float maxDistanceSqr)
    {
        return (a - b).sqrMagnitude <= maxDistanceSqr;
    }

    sealed class RecentDetectionCentroid
    {
        public readonly Vector3 WorldPosition;
        public Coroutine ExpiryCoroutine;

        public RecentDetectionCentroid(Vector3 worldPosition)
        {
            WorldPosition = worldPosition;
        }
    }
}

public sealed class MarkerManagerCoroutineHost : MonoBehaviour
{
}
