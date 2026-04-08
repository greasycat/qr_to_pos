using System.Collections.Generic;
using UnityEngine;

public sealed class MarkerConstructionManager
{
    const float MinimumWallLength = 0.001f;
    static readonly Color DefaultWallColor = new Color(0.78f, 0.78f, 0.78f, 1f);

    readonly Dictionary<string, TrackedWallPoint> trackedWallPoints = new Dictionary<string, TrackedWallPoint>();
    readonly Dictionary<string, WallConstructionState> wallStates = new Dictionary<string, WallConstructionState>();
    Material wallMaterial;

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
        if (wallDirection.sqrMagnitude < MinimumWallLength * MinimumWallLength)
            wallDirection = Vector3.forward;

        Vector3 wallLengthDirection = wallDirection.normalized;
        Vector3 wallThicknessDirection = Vector3.Cross(Vector3.up, wallLengthDirection);
        if (wallThicknessDirection.sqrMagnitude < MinimumWallLength * MinimumWallLength)
            wallThicknessDirection = Vector3.right;
        else
            wallThicknessDirection.Normalize();

        segment.name = string.Format("Construction_{0}_Wall_{1}", displayName, segmentIndex);
        segment.transform.SetParent(markerParent, false);
        segment.transform.localPosition = Vector3.zero;
        segment.transform.localRotation = Quaternion.identity;
        segment.transform.localScale = Vector3.one;

        Vector3 halfThicknessOffset = wallThicknessDirection * (wallThickness * 0.5f);
        Vector3 heightOffset = Vector3.up * wallHeight;

        Vector3[] wallCorners = new Vector3[8];
        wallCorners[0] = leveledStart - halfThicknessOffset;
        wallCorners[1] = leveledStart + halfThicknessOffset;
        wallCorners[2] = leveledEnd + halfThicknessOffset;
        wallCorners[3] = leveledEnd - halfThicknessOffset;
        wallCorners[4] = wallCorners[0] + heightOffset;
        wallCorners[5] = wallCorners[1] + heightOffset;
        wallCorners[6] = wallCorners[2] + heightOffset;
        wallCorners[7] = wallCorners[3] + heightOffset;

        UpdateWallMesh(
            segment,
            markerParent,
            wallCorners,
            wallLengthDirection,
            wallThicknessDirection);
    }

    static int CompareTrackedWallPoints(TrackedWallPoint left, TrackedWallPoint right)
    {
        int orderComparison = left.Order.CompareTo(right.Order);
        if (orderComparison != 0)
            return orderComparison;

        return string.CompareOrdinal(left.TagKey, right.TagKey);
    }

    void UpdateWallMesh(
        GameObject segment,
        Transform markerParent,
        Vector3[] wallCorners,
        Vector3 wallLengthDirection,
        Vector3 wallThicknessDirection)
    {
        if (segment == null || wallCorners == null || wallCorners.Length < 8)
            return;

        MeshFilter meshFilter = segment.GetComponent<MeshFilter>();
        if (meshFilter == null)
            meshFilter = segment.AddComponent<MeshFilter>();

        MeshRenderer meshRenderer = segment.GetComponent<MeshRenderer>();
        if (meshRenderer == null)
            meshRenderer = segment.AddComponent<MeshRenderer>();
        meshRenderer.sharedMaterial = GetOrCreateWallMaterial();

        MeshCollider meshCollider = segment.GetComponent<MeshCollider>();
        if (meshCollider == null)
            meshCollider = segment.AddComponent<MeshCollider>();

        Mesh wallMesh = meshFilter.sharedMesh;
        if (wallMesh == null)
        {
            wallMesh = new Mesh();
            wallMesh.name = "MarkerConstructionWallMesh";
            meshFilter.sharedMesh = wallMesh;
        }
        else
        {
            wallMesh.Clear();
        }

        Vector3[] localCorners = new Vector3[wallCorners.Length];
        for (int i = 0; i < wallCorners.Length; i++)
            localCorners[i] = ToLocalPoint(markerParent, wallCorners[i]);

        Vector3[] vertices;
        int[] triangles;
        Vector2[] uvs;
        Vector3 localUp = ToLocalDirection(markerParent, Vector3.up);
        Vector3 localLengthDirection = ToLocalDirection(markerParent, wallLengthDirection);
        Vector3 localThicknessDirection = ToLocalDirection(markerParent, wallThicknessDirection);
        BuildWallGeometry(
            localCorners,
            localUp,
            localLengthDirection,
            localThicknessDirection,
            out vertices,
            out triangles,
            out uvs);

        wallMesh.vertices = vertices;
        wallMesh.triangles = triangles;
        wallMesh.uv = uvs;
        wallMesh.RecalculateBounds();
        wallMesh.RecalculateNormals();

        meshCollider.sharedMesh = null;
        meshCollider.sharedMesh = wallMesh;
    }

    void BuildWallGeometry(
        Vector3[] wallCorners,
        Vector3 localUp,
        Vector3 localLengthDirection,
        Vector3 localThicknessDirection,
        out Vector3[] vertices,
        out int[] triangles,
        out Vector2[] uvs)
    {
        var vertexList = new List<Vector3>(24);
        var triangleList = new List<int>(36);
        var uvList = new List<Vector2>(24);

        AddFace(vertexList, triangleList, uvList, wallCorners[0], wallCorners[1], wallCorners[2], wallCorners[3], -localUp);
        AddFace(vertexList, triangleList, uvList, wallCorners[4], wallCorners[5], wallCorners[6], wallCorners[7], localUp);
        AddFace(vertexList, triangleList, uvList, wallCorners[0], wallCorners[1], wallCorners[5], wallCorners[4], -localLengthDirection);
        AddFace(vertexList, triangleList, uvList, wallCorners[3], wallCorners[2], wallCorners[6], wallCorners[7], localLengthDirection);
        AddFace(vertexList, triangleList, uvList, wallCorners[0], wallCorners[4], wallCorners[7], wallCorners[3], -localThicknessDirection);
        AddFace(vertexList, triangleList, uvList, wallCorners[1], wallCorners[2], wallCorners[6], wallCorners[5], localThicknessDirection);

        vertices = vertexList.ToArray();
        triangles = triangleList.ToArray();
        uvs = uvList.ToArray();
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
        segment.AddComponent<MeshFilter>();
        MeshRenderer meshRenderer = segment.AddComponent<MeshRenderer>();
        meshRenderer.sharedMaterial = GetOrCreateWallMaterial();
        segment.AddComponent<MeshCollider>();
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

    static Vector3 ToLocalPoint(Transform markerParent, Vector3 worldPoint)
    {
        if (markerParent == null)
            return worldPoint;

        return markerParent.InverseTransformPoint(worldPoint);
    }

    static Vector3 ToLocalDirection(Transform markerParent, Vector3 worldDirection)
    {
        if (markerParent == null)
            return worldDirection.normalized;

        return markerParent.InverseTransformDirection(worldDirection).normalized;
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
