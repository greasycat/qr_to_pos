using System.Collections.Generic;
using UnityEngine;

public sealed class MarkerConstructionManager
{
    const float MinimumWallLength = 0.001f;

    readonly Dictionary<string, TrackedWallPoint> trackedWallPoints = new Dictionary<string, TrackedWallPoint>();
    readonly Dictionary<string, WallConstructionState> wallStates = new Dictionary<string, WallConstructionState>();

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
            trackedPoint.Position = targetPosition;
        }
        else
        {
            trackedPoint.Position = Vector3.MoveTowards(
                trackedPoint.Position,
                targetPosition,
                MarkerManager.MaxMarkerMovementDistancePerUpdate);
        }

        trackedPoint.BindingKey = assignment.BindingKey;
        trackedPoint.DisplayName = assignment.DisplayName;
        trackedPoint.TagKey = tagKey;
        trackedPoint.Order = assignment.Order;
        trackedPoint.Binding = assignment.Binding;
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
            ResolvedWallAxes axes = ResolveWallAxes(wallSettings);

            WallConstructionState wallState;
            if (!wallStates.TryGetValue(entry.Key, out wallState))
            {
                wallState = new WallConstructionState();
                wallStates[entry.Key] = wallState;
            }

            if (!axes.IsValid && !wallState.HasLoggedInvalidAxesWarning)
            {
                Debug.LogWarningFormat(
                    "MarkerConstructionManager: Wall construction '{0}' must use distinct length, height, and thickness axes. Falling back to Z/Y/X.",
                    firstPoint.DisplayName);
                wallState.HasLoggedInvalidAxesWarning = true;
            }
            else if (axes.IsValid)
            {
                wallState.HasLoggedInvalidAxesWarning = false;
            }

            GameObject segmentPrefab = wallSettings != null ? wallSettings.segmentPrefab : null;
            if (wallState.SegmentPrefab != segmentPrefab)
            {
                DestroySegments(wallState);
                wallState.SegmentPrefab = segmentPrefab;
            }

            EnsureSegmentCapacity(wallState, points.Count - 1, segmentPrefab);
            for (int i = 0; i < points.Count - 1; i++)
            {
                GameObject segment = wallState.Segments[i];
                if (segment == null)
                {
                    segment = CreateSegment(segmentPrefab);
                    wallState.Segments[i] = segment;
                }

                UpdateWallSegment(
                    segment,
                    markerParent,
                    firstPoint.DisplayName,
                    i,
                    points[i].Position,
                    points[i + 1].Position,
                    wallSettings,
                    axes);
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

    void EnsureSegmentCapacity(WallConstructionState wallState, int requiredCount, GameObject segmentPrefab)
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
            wallState.Segments.Add(CreateSegment(segmentPrefab));
    }

    void UpdateWallSegment(
        GameObject segment,
        Transform markerParent,
        string displayName,
        int segmentIndex,
        Vector3 startPoint,
        Vector3 endPoint,
        MarkerWallConstructionSettings wallSettings,
        ResolvedWallAxes axes)
    {
        if (segment == null)
            return;

        float wallHeight = Mathf.Max(wallSettings != null ? wallSettings.height : 1f, MinimumWallLength);
        float wallThickness = Mathf.Max(wallSettings != null ? wallSettings.thickness : 0.2f, MinimumWallLength);
        Vector3 wallDirection = Vector3.ProjectOnPlane(endPoint - startPoint, Vector3.up);
        if (wallDirection.sqrMagnitude < MinimumWallLength * MinimumWallLength)
            wallDirection = Vector3.forward;

        float wallLength = Mathf.Max(wallDirection.magnitude, MinimumWallLength);
        Vector3 worldLengthAxis = wallDirection.normalized;
        Vector3 worldHeightAxis = Vector3.up;
        Vector3 worldThicknessAxis = UsesRightHandedBasis(
            axes.LocalLengthAxis,
            axes.LocalHeightAxis,
            axes.LocalThicknessAxis)
            ? Vector3.Cross(worldLengthAxis, worldHeightAxis)
            : Vector3.Cross(worldHeightAxis, worldLengthAxis);
        if (worldThicknessAxis.sqrMagnitude < MinimumWallLength * MinimumWallLength)
            worldThicknessAxis = Vector3.right;
        else
            worldThicknessAxis.Normalize();

        segment.name = string.Format("Construction_{0}_Wall_{1}", displayName, segmentIndex);
        segment.transform.SetParent(markerParent, true);
        segment.transform.rotation = BuildRotation(
            axes.LocalLengthAxis,
            axes.LocalHeightAxis,
            axes.LocalThicknessAxis,
            worldLengthAxis,
            worldHeightAxis,
            worldThicknessAxis);
        segment.transform.localScale = BuildScale(axes, wallLength, wallHeight, wallThickness);
        segment.transform.position = Vector3.Lerp(startPoint, endPoint, 0.5f) + Vector3.up * (wallHeight * 0.5f);
        AlignBottomToSurface(segment, Mathf.Max(startPoint.y, endPoint.y));
    }

    static int CompareTrackedWallPoints(TrackedWallPoint left, TrackedWallPoint right)
    {
        int orderComparison = left.Order.CompareTo(right.Order);
        if (orderComparison != 0)
            return orderComparison;

        return string.CompareOrdinal(left.TagKey, right.TagKey);
    }

    static void AlignBottomToSurface(GameObject segment, float targetSurfaceY)
    {
        Bounds bounds;
        if (!MarkerManager.TryGetObjectBounds(segment, out bounds))
            return;

        Vector3 position = segment.transform.position;
        position.y = targetSurfaceY + (segment.transform.position.y - bounds.min.y);
        segment.transform.position = position;
    }

    static GameObject CreateSegment(GameObject segmentPrefab)
    {
        return segmentPrefab != null
            ? Object.Instantiate(segmentPrefab)
            : GameObject.CreatePrimitive(PrimitiveType.Cube);
    }

    static Quaternion BuildRotation(
        Vector3 localLengthAxis,
        Vector3 localHeightAxis,
        Vector3 localThicknessAxis,
        Vector3 worldLengthAxis,
        Vector3 worldHeightAxis,
        Vector3 worldThicknessAxis)
    {
        Matrix4x4 localBasis = BuildBasisMatrix(localLengthAxis, localHeightAxis, localThicknessAxis);
        Matrix4x4 worldBasis = BuildBasisMatrix(worldLengthAxis, worldHeightAxis, worldThicknessAxis);
        Matrix4x4 rotationMatrix = worldBasis * localBasis.transpose;
        return rotationMatrix.rotation;
    }

    static Matrix4x4 BuildBasisMatrix(Vector3 firstAxis, Vector3 secondAxis, Vector3 thirdAxis)
    {
        Matrix4x4 basis = Matrix4x4.identity;
        basis.SetColumn(0, new Vector4(firstAxis.x, firstAxis.y, firstAxis.z, 0f));
        basis.SetColumn(1, new Vector4(secondAxis.x, secondAxis.y, secondAxis.z, 0f));
        basis.SetColumn(2, new Vector4(thirdAxis.x, thirdAxis.y, thirdAxis.z, 0f));
        return basis;
    }

    static Vector3 BuildScale(ResolvedWallAxes axes, float wallLength, float wallHeight, float wallThickness)
    {
        Vector3 scale = Vector3.one;
        ApplyAxisScale(ref scale, axes.LengthDirection, wallLength);
        ApplyAxisScale(ref scale, axes.HeightDirection, wallHeight);
        ApplyAxisScale(ref scale, axes.ThicknessDirection, wallThickness);
        return scale;
    }

    static void ApplyAxisScale(ref Vector3 scale, MarkerConstructionAxisDirection axisDirection, float value)
    {
        switch (axisDirection)
        {
            case MarkerConstructionAxisDirection.PositiveX:
            case MarkerConstructionAxisDirection.NegativeX:
                scale.x = value;
                break;
            case MarkerConstructionAxisDirection.PositiveY:
            case MarkerConstructionAxisDirection.NegativeY:
                scale.y = value;
                break;
            case MarkerConstructionAxisDirection.PositiveZ:
            case MarkerConstructionAxisDirection.NegativeZ:
                scale.z = value;
                break;
        }
    }

    static ResolvedWallAxes ResolveWallAxes(MarkerWallConstructionSettings wallSettings)
    {
        MarkerConstructionAxisDirection lengthDirection = wallSettings != null
            ? wallSettings.lengthAxis
            : MarkerConstructionAxisDirection.PositiveZ;
        MarkerConstructionAxisDirection heightDirection = wallSettings != null
            ? wallSettings.heightAxis
            : MarkerConstructionAxisDirection.PositiveY;
        MarkerConstructionAxisDirection thicknessDirection = wallSettings != null
            ? wallSettings.thicknessAxis
            : MarkerConstructionAxisDirection.PositiveX;

        if (!UsesDistinctAxes(lengthDirection, heightDirection, thicknessDirection))
        {
            lengthDirection = MarkerConstructionAxisDirection.PositiveZ;
            heightDirection = MarkerConstructionAxisDirection.PositiveY;
            thicknessDirection = MarkerConstructionAxisDirection.PositiveX;
            return new ResolvedWallAxes(
                false,
                lengthDirection,
                heightDirection,
                thicknessDirection,
                GetAxisVector(lengthDirection),
                GetAxisVector(heightDirection),
                GetAxisVector(thicknessDirection));
        }

        return new ResolvedWallAxes(
            true,
            lengthDirection,
            heightDirection,
            thicknessDirection,
            GetAxisVector(lengthDirection),
            GetAxisVector(heightDirection),
            GetAxisVector(thicknessDirection));
    }

    static bool UsesDistinctAxes(
        MarkerConstructionAxisDirection lengthDirection,
        MarkerConstructionAxisDirection heightDirection,
        MarkerConstructionAxisDirection thicknessDirection)
    {
        int lengthAxisIndex = GetAxisIndex(lengthDirection);
        int heightAxisIndex = GetAxisIndex(heightDirection);
        int thicknessAxisIndex = GetAxisIndex(thicknessDirection);

        return lengthAxisIndex != heightAxisIndex
            && lengthAxisIndex != thicknessAxisIndex
            && heightAxisIndex != thicknessAxisIndex;
    }

    static int GetAxisIndex(MarkerConstructionAxisDirection axisDirection)
    {
        switch (axisDirection)
        {
            case MarkerConstructionAxisDirection.PositiveX:
            case MarkerConstructionAxisDirection.NegativeX:
                return 0;
            case MarkerConstructionAxisDirection.PositiveY:
            case MarkerConstructionAxisDirection.NegativeY:
                return 1;
            default:
                return 2;
        }
    }

    static Vector3 GetAxisVector(MarkerConstructionAxisDirection axisDirection)
    {
        switch (axisDirection)
        {
            case MarkerConstructionAxisDirection.PositiveX:
                return Vector3.right;
            case MarkerConstructionAxisDirection.NegativeX:
                return Vector3.left;
            case MarkerConstructionAxisDirection.PositiveY:
                return Vector3.up;
            case MarkerConstructionAxisDirection.NegativeY:
                return Vector3.down;
            case MarkerConstructionAxisDirection.PositiveZ:
                return Vector3.forward;
            default:
                return Vector3.back;
        }
    }

    static bool UsesRightHandedBasis(Vector3 firstAxis, Vector3 secondAxis, Vector3 thirdAxis)
    {
        return Vector3.Dot(Vector3.Cross(firstAxis, secondAxis), thirdAxis) > 0f;
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
        public GameObject SegmentPrefab;
        public bool HasLoggedInvalidAxesWarning;
    }

    struct ResolvedWallAxes
    {
        public readonly bool IsValid;
        public readonly MarkerConstructionAxisDirection LengthDirection;
        public readonly MarkerConstructionAxisDirection HeightDirection;
        public readonly MarkerConstructionAxisDirection ThicknessDirection;
        public readonly Vector3 LocalLengthAxis;
        public readonly Vector3 LocalHeightAxis;
        public readonly Vector3 LocalThicknessAxis;

        public ResolvedWallAxes(
            bool isValid,
            MarkerConstructionAxisDirection lengthDirection,
            MarkerConstructionAxisDirection heightDirection,
            MarkerConstructionAxisDirection thicknessDirection,
            Vector3 localLengthAxis,
            Vector3 localHeightAxis,
            Vector3 localThicknessAxis)
        {
            IsValid = isValid;
            LengthDirection = lengthDirection;
            HeightDirection = heightDirection;
            ThicknessDirection = thicknessDirection;
            LocalLengthAxis = localLengthAxis;
            LocalHeightAxis = localHeightAxis;
            LocalThicknessAxis = localThicknessAxis;
        }
    }
}
