using System.Collections.Generic;
using UnityEngine;

public sealed class QRTerrainMapper
{
    readonly Terrain terrain;
    readonly bool flipX;
    readonly bool flipZ;

    public QRTerrainMapper(Terrain terrain, bool flipX, bool flipZ)
    {
        this.terrain = terrain;
        this.flipX = flipX;
        this.flipZ = flipZ;
    }

    public bool TryGetMarkerPosition(QRDetection detection, float markerVerticalOffset, out Vector3 worldPosition, out bool isOutOfBounds)
    {
        worldPosition = Vector3.zero;
        isOutOfBounds = false;

        if (terrain == null)
            return false;

        float xNorm;
        float zNorm;
        if (!TryGetDetectionNormalizedPosition(detection, out xNorm, out zNorm, out isOutOfBounds))
            return false;

        worldPosition = GetMarkerWorldPosition(xNorm, zNorm, markerVerticalOffset);
        return true;
    }

    public List<QRDebugMarkerPlacement> GetDebugBounds(float debugBoundsY)
    {
        var markers = new List<QRDebugMarkerPlacement>(4);
        if (terrain == null)
            return markers;

        markers.Add(new QRDebugMarkerPlacement("QRBounds_MinMin", GetDebugWorldPosition(0f, 0f, debugBoundsY)));
        markers.Add(new QRDebugMarkerPlacement("QRBounds_MinMax", GetDebugWorldPosition(0f, 1f, debugBoundsY)));
        markers.Add(new QRDebugMarkerPlacement("QRBounds_MaxMin", GetDebugWorldPosition(1f, 0f, debugBoundsY)));
        markers.Add(new QRDebugMarkerPlacement("QRBounds_MaxMax", GetDebugWorldPosition(1f, 1f, debugBoundsY)));
        return markers;
    }

    bool TryGetDetectionNormalizedPosition(QRDetection detection, out float xNorm, out float zNorm, out bool isOutOfBounds)
    {
        xNorm = 0f;
        zNorm = 0f;
        isOutOfBounds = false;

        if (detection.depth_centroid_pct == null || detection.depth_centroid_pct.Length < 2)
            return false;

        float centroidXPct = detection.depth_centroid_pct[0];
        float centroidYPct = detection.depth_centroid_pct[1];
        if (float.IsNaN(centroidXPct) || float.IsInfinity(centroidXPct)
            || float.IsNaN(centroidYPct) || float.IsInfinity(centroidYPct))
            return false;

        isOutOfBounds = centroidXPct < 0f || centroidXPct > 100f
            || centroidYPct < 0f || centroidYPct > 100f;

        // TerrainEditor writes the cropped depth image across the full terrain heightmap.
        xNorm = centroidYPct / 100f;
        zNorm = centroidXPct / 100f;

        if (flipX)
            xNorm = 1f - xNorm;
        if (flipZ)
            zNorm = 1f - zNorm;
        return true;
    }

    Vector3 GetMarkerWorldPosition(float xNorm, float zNorm, float markerVerticalOffset)
    {
        TerrainData terrainData = terrain.terrainData;
        Vector3 terrainSize = terrainData.size;
        Vector3 terrainPosition = terrain.GetPosition();

        float worldX = terrainPosition.x + xNorm * terrainSize.x;
        float worldZ = terrainPosition.z + zNorm * terrainSize.z;
        float worldY = terrain.SampleHeight(new Vector3(worldX, 0f, worldZ)) + terrainPosition.y + markerVerticalOffset;
        return new Vector3(worldX, worldY, worldZ);
    }

    Vector3 GetDebugWorldPosition(float xNorm, float zNorm, float debugBoundsY)
    {
        if (flipX)
            xNorm = 1f - xNorm;
        if (flipZ)
            zNorm = 1f - zNorm;

        TerrainData terrainData = terrain.terrainData;
        Vector3 terrainSize = terrainData.size;
        Vector3 terrainPosition = terrain.GetPosition();

        float worldX = terrainPosition.x + xNorm * terrainSize.x;
        float worldZ = terrainPosition.z + zNorm * terrainSize.z;
        return new Vector3(worldX, debugBoundsY, worldZ);
    }
}
