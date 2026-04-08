using System.Collections.Generic;
using UnityEngine;

public sealed class MarkerConstructionManager
{
    const float MinimumWallLength = 0.001f;
    static readonly Color DefaultWallColor = new Color(0.78f, 0.78f, 0.78f, 1f);

    readonly Dictionary<string, TrackedWallPoint> trackedWallPoints = new Dictionary<string, TrackedWallPoint>();
    readonly Dictionary<string, WallConstructionState> wallStates = new Dictionary<string, WallConstructionState>();
    Material wallMaterial;
    Mesh wallSegmentMesh;

    public void TrackWallPoint(Vector3 targetPosition, MarkerDetection detection, MarkerConstructionAssignment assignment)
    {
        if (assignment == null || assignment.Binding == null)
            return;

        string tagKey;
        if (!MarkerManager.TryGetTrackingKey(detection, out tagKey))
            return;

        string pointKey = string.Format("{0}:{1}", assignment.BindingKey, tagKey);
        TrackedWallPoint trackedPoint;
        if (!trackedWallPoints.TryGetValue(pointKey, out trackedPoint))
        {
            trackedPoint = new TrackedWallPoint();
            trackedWallPoints[pointKey] = trackedPoint;
        }

        trackedPoint.BindingKey = assignment.BindingKey;
        trackedPoint.DisplayName = assignment.DisplayName;
        trackedPoint.TagKey = tagKey;
        trackedPoint.Order = assignment.Order;
        trackedPoint.Binding = assignment.Binding;
        trackedPoint.Position = targetPosition;
        trackedPoint.LastSeenTime = Time.time;
    }

    public void SpawnWallFromSnapshot(Transform markerParent, List<MarkerConstructionPointSnapshot> pointSnapshots, float seenTime)
    {
        if (pointSnapshots == null || pointSnapshots.Count < 2)
            return;

        for (int i = 0; i < pointSnapshots.Count; i++)
        {
            MarkerConstructionPointSnapshot snapshot = pointSnapshots[i];
            if (snapshot == null || snapshot.Binding == null)
                continue;

            string pointKey = string.Format("{0}:{1}", snapshot.BindingKey, snapshot.TagKey);
            TrackedWallPoint trackedPoint;
            if (!trackedWallPoints.TryGetValue(pointKey, out trackedPoint))
            {
                trackedPoint = new TrackedWallPoint();
                trackedWallPoints[pointKey] = trackedPoint;
            }

            trackedPoint.BindingKey = snapshot.BindingKey;
            trackedPoint.DisplayName = snapshot.DisplayName;
            trackedPoint.TagKey = snapshot.TagKey;
            trackedPoint.Order = snapshot.Order;
            trackedPoint.Binding = snapshot.Binding;
            trackedPoint.Position = snapshot.Position;
            trackedPoint.LastSeenTime = seenTime;
        }

        RebuildWalls(markerParent);
    }

    public bool HasWallConstruction(string bindingKey)
    {
        if (string.IsNullOrEmpty(bindingKey))
            return false;

        WallConstructionState wallState;
        if (!wallStates.TryGetValue(bindingKey, out wallState) || wallState == null)
            return false;

        return wallState.Segments.Count > 0;
    }

    public void RefreshTrackedConstructions(Transform markerParent)
    {
        RebuildWalls(markerParent);
    }

    public void PruneExpiredConstructions(Transform markerParent, float currentTime)
    {
        if (trackedWallPoints.Count == 0)
            return;

        var expiredKeys = new List<string>();
        foreach (var entry in trackedWallPoints)
        {
            TrackedWallPoint trackedPoint = entry.Value;
            if (trackedPoint == null)
            {
                expiredKeys.Add(entry.Key);
                continue;
            }

            if (currentTime - trackedPoint.LastSeenTime > MarkerManager.MarkerLifetimeSeconds)
                expiredKeys.Add(entry.Key);
        }

        if (expiredKeys.Count == 0)
            return;

        for (int i = 0; i < expiredKeys.Count; i++)
            trackedWallPoints.Remove(expiredKeys[i]);

        RebuildWalls(markerParent);
    }

    public void ClearAll()
    {
        foreach (var entry in wallStates)
            DestroySegments(entry.Value);

        trackedWallPoints.Clear();
        wallStates.Clear();

        if (wallMaterial != null)
        {
            Object.Destroy(wallMaterial);
            wallMaterial = null;
        }

        if (wallSegmentMesh != null)
        {
            Object.Destroy(wallSegmentMesh);
            wallSegmentMesh = null;
        }
    }

    void RebuildWalls(Transform markerParent)
    {
        var groupedPoints = new Dictionary<string, List<TrackedWallPoint>>();
        foreach (var entry in trackedWallPoints)
        {
            TrackedWallPoint trackedPoint = entry.Value;
            if (trackedPoint == null || trackedPoint.Binding == null || trackedPoint.Binding.choice != MarkerConstructionChoice.Wall)
                continue;

            List<TrackedWallPoint> points;
            if (!groupedPoints.TryGetValue(trackedPoint.BindingKey, out points))
            {
                points = new List<TrackedWallPoint>();
                groupedPoints[trackedPoint.BindingKey] = points;
            }

            points.Add(trackedPoint);
        }

        var activeGroups = new HashSet<string>();
        foreach (var entry in groupedPoints)
        {
            List<TrackedWallPoint> points = entry.Value;
            points.Sort(CompareTrackedWallPoints);

            if (points.Count < 2)
            {
                ClearWallState(entry.Key);
                continue;
            }

            TrackedWallPoint firstPoint = points[0];
            MarkerWallConstructionSettings wallSettings = firstPoint.Binding.wall;

            WallConstructionState wallState;
            if (!wallStates.TryGetValue(entry.Key, out wallState))
            {
                wallState = new WallConstructionState();
                wallStates[entry.Key] = wallState;
            }

            EnsureSegmentCapacity(wallState, points.Count - 1);
            for (int i = 0; i < points.Count - 1; i++)
            {
                GameObject segment = wallState.Segments[i];
                if (segment == null)
                {
                    segment = CreateSegment();
                    wallState.Segments[i] = segment;
                }

                UpdateWallSegment(
                    segment,
                    markerParent,
                    firstPoint.DisplayName,
                    i,
                    points[i].Position,
                    points[i + 1].Position,
                    wallSettings);
            }

            activeGroups.Add(entry.Key);
        }

        var staleGroups = new List<string>();
        foreach (var entry in wallStates)
        {
            if (!activeGroups.Contains(entry.Key))
            {
                DestroySegments(entry.Value);
                staleGroups.Add(entry.Key);
            }
        }

        for (int i = 0; i < staleGroups.Count; i++)
            wallStates.Remove(staleGroups[i]);
    }

    void EnsureSegmentCapacity(WallConstructionState wallState, int requiredCount)
    {
        while (wallState.Segments.Count > requiredCount)
        {
            int lastIndex = wallState.Segments.Count - 1;
            GameObject segment = wallState.Segments[lastIndex];
            if (segment != null)
                Object.Destroy(segment);

            wallState.Segments.RemoveAt(lastIndex);
        }

        while (wallState.Segments.Count < requiredCount)
            wallState.Segments.Add(CreateSegment());
    }

    void UpdateWallSegment(
        GameObject segment,
        Transform markerParent,
        string displayName,
        int segmentIndex,
        Vector3 startPoint,
        Vector3 endPoint,
        MarkerWallConstructionSettings wallSettings)
    {
        if (segment == null)
            return;

        float wallHeight = Mathf.Max(wallSettings != null ? wallSettings.height : 1f, MinimumWallLength);
        float wallThickness = Mathf.Max(wallSettings != null ? wallSettings.thickness : 0.2f, MinimumWallLength);
        float wallBaseY = Mathf.Max(startPoint.y, endPoint.y);
        Vector3 leveledStart = new Vector3(startPoint.x, wallBaseY, startPoint.z);
        Vector3 leveledEnd = new Vector3(endPoint.x, wallBaseY, endPoint.z);
        Vector3 wallDirection = leveledEnd - leveledStart;
        float wallLength = wallDirection.magnitude;
        if (wallLength < MinimumWallLength)
        {
            wallDirection = Vector3.forward;
            wallLength = MinimumWallLength;
        }

        Vector3 wallLengthDirection = wallDirection.normalized;
        Vector3 wallCenter = Vector3.Lerp(leveledStart, leveledEnd, 0.5f);

        segment.name = string.Format("Construction_{0}_Wall_{1}", displayName, segmentIndex);
        segment.transform.SetParent(markerParent, false);
        segment.transform.position = wallCenter;
        segment.transform.rotation = Quaternion.LookRotation(wallLengthDirection, Vector3.up);
        SetWorldScale(segment.transform, markerParent, new Vector3(wallThickness, wallHeight, wallLength));
    }

    static int CompareTrackedWallPoints(TrackedWallPoint left, TrackedWallPoint right)
    {
        int orderComparison = left.Order.CompareTo(right.Order);
        if (orderComparison != 0)
            return orderComparison;

        return string.CompareOrdinal(left.TagKey, right.TagKey);
    }

    static void SetWorldScale(Transform segmentTransform, Transform markerParent, Vector3 worldScale)
    {
        if (segmentTransform == null)
            return;

        if (markerParent == null)
        {
            segmentTransform.localScale = worldScale;
            return;
        }

        Vector3 parentScale = markerParent.lossyScale;
        segmentTransform.localScale = new Vector3(
            DivideScale(worldScale.x, parentScale.x),
            DivideScale(worldScale.y, parentScale.y),
            DivideScale(worldScale.z, parentScale.z));
    }

    static float DivideScale(float value, float divisor)
    {
        return Mathf.Approximately(divisor, 0f) ? value : value / divisor;
    }

    Mesh GetOrCreateWallSegmentMesh()
    {
        if (wallSegmentMesh != null)
            return wallSegmentMesh;

        wallSegmentMesh = new Mesh();
        wallSegmentMesh.name = "MarkerConstructionWallSegmentMesh";

        var vertexList = new List<Vector3>(24);
        var triangleList = new List<int>(36);
        var uvList = new List<Vector2>(24);

        Vector3[] corners = new Vector3[8];
        corners[0] = new Vector3(-0.5f, 0f, -0.5f);
        corners[1] = new Vector3(0.5f, 0f, -0.5f);
        corners[2] = new Vector3(0.5f, 0f, 0.5f);
        corners[3] = new Vector3(-0.5f, 0f, 0.5f);
        corners[4] = new Vector3(-0.5f, 1f, -0.5f);
        corners[5] = new Vector3(0.5f, 1f, -0.5f);
        corners[6] = new Vector3(0.5f, 1f, 0.5f);
        corners[7] = new Vector3(-0.5f, 1f, 0.5f);

        AddFace(vertexList, triangleList, uvList, corners[0], corners[1], corners[2], corners[3], Vector3.down);
        AddFace(vertexList, triangleList, uvList, corners[4], corners[5], corners[6], corners[7], Vector3.up);
        AddFace(vertexList, triangleList, uvList, corners[0], corners[1], corners[5], corners[4], Vector3.back);
        AddFace(vertexList, triangleList, uvList, corners[3], corners[2], corners[6], corners[7], Vector3.forward);
        AddFace(vertexList, triangleList, uvList, corners[0], corners[4], corners[7], corners[3], Vector3.left);
        AddFace(vertexList, triangleList, uvList, corners[1], corners[2], corners[6], corners[5], Vector3.right);

        wallSegmentMesh.vertices = vertexList.ToArray();
        wallSegmentMesh.triangles = triangleList.ToArray();
        wallSegmentMesh.uv = uvList.ToArray();
        wallSegmentMesh.RecalculateBounds();
        wallSegmentMesh.RecalculateNormals();
        return wallSegmentMesh;
    }

    static void AddFace(
        List<Vector3> vertexList,
        List<int> triangleList,
        List<Vector2> uvList,
        Vector3 a,
        Vector3 b,
        Vector3 c,
        Vector3 d,
        Vector3 desiredNormal)
    {
        int baseIndex = vertexList.Count;
        vertexList.Add(a);
        vertexList.Add(b);
        vertexList.Add(c);
        vertexList.Add(d);

        uvList.Add(new Vector2(0f, 0f));
        uvList.Add(new Vector2(1f, 0f));
        uvList.Add(new Vector2(1f, 1f));
        uvList.Add(new Vector2(0f, 1f));

        Vector3 actualNormal = Vector3.Cross(b - a, c - a);
        bool windingMatches = Vector3.Dot(actualNormal, desiredNormal) >= 0f;
        if (windingMatches)
        {
            triangleList.Add(baseIndex + 0);
            triangleList.Add(baseIndex + 1);
            triangleList.Add(baseIndex + 2);
            triangleList.Add(baseIndex + 0);
            triangleList.Add(baseIndex + 2);
            triangleList.Add(baseIndex + 3);
            return;
        }

        triangleList.Add(baseIndex + 0);
        triangleList.Add(baseIndex + 2);
        triangleList.Add(baseIndex + 1);
        triangleList.Add(baseIndex + 0);
        triangleList.Add(baseIndex + 3);
        triangleList.Add(baseIndex + 2);
    }

    GameObject CreateSegment()
    {
        var segment = new GameObject();
        MeshFilter meshFilter = segment.AddComponent<MeshFilter>();
        meshFilter.sharedMesh = GetOrCreateWallSegmentMesh();
        MeshRenderer meshRenderer = segment.AddComponent<MeshRenderer>();
        meshRenderer.sharedMaterial = GetOrCreateWallMaterial();
        MeshCollider meshCollider = segment.AddComponent<MeshCollider>();
        meshCollider.sharedMesh = GetOrCreateWallSegmentMesh();
        return segment;
    }

    Material GetOrCreateWallMaterial()
    {
        if (wallMaterial != null)
            return wallMaterial;

        Shader shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null)
            shader = Shader.Find("Standard");
        if (shader == null)
            shader = Shader.Find("Unlit/Color");
        if (shader == null)
            shader = Shader.Find("Sprites/Default");
        if (shader == null)
        {
            Debug.LogWarning("MarkerConstructionManager: Could not find a wall shader. Procedural walls will render without an assigned material.");
            return null;
        }

        wallMaterial = new Material(shader);
        wallMaterial.name = "MarkerConstructionWallMaterial";
        wallMaterial.color = DefaultWallColor;
        return wallMaterial;
    }

    void ClearWallState(string bindingKey)
    {
        WallConstructionState wallState;
        if (!wallStates.TryGetValue(bindingKey, out wallState))
            return;

        DestroySegments(wallState);
        wallStates.Remove(bindingKey);
    }

    static void DestroySegments(WallConstructionState wallState)
    {
        if (wallState == null)
            return;

        for (int i = 0; i < wallState.Segments.Count; i++)
        {
            if (wallState.Segments[i] != null)
                Object.Destroy(wallState.Segments[i]);
        }

        wallState.Segments.Clear();
    }

    sealed class TrackedWallPoint
    {
        public string BindingKey;
        public string DisplayName;
        public string TagKey;
        public int Order;
        public MarkerConstructionBinding Binding;
        public Vector3 Position;
        public float LastSeenTime;
    }

    sealed class WallConstructionState
    {
        public readonly List<GameObject> Segments = new List<GameObject>();
    }
}
