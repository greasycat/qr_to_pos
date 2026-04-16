using System.Collections.Generic;
using UnityEngine;

public sealed class MarkerConstructionManager
{
    const float MinimumWallLength = 0.001f;
    static readonly Color DefaultWallColor = new Color(0.78f, 0.78f, 0.78f, 1f);

    readonly HashSet<string> activeWallBindings = new HashSet<string>();
    readonly Dictionary<string, WallConstructionState> wallStates = new Dictionary<string, WallConstructionState>();
    Material wallMaterial;
    Mesh wallSegmentMesh;

    public void BeginWallRefresh()
    {
        activeWallBindings.Clear();
    }

    public void UpdateWall(
        Transform markerParent,
        string bindingKey,
        string displayName,
        MarkerWallConstructionSettings wallSettings,
        Vector3[] orderedPoints)
    {
        if (string.IsNullOrEmpty(bindingKey))
            return;

        if (orderedPoints == null || orderedPoints.Length < 2)
        {
            ClearWallState(bindingKey);
            return;
        }

        WallConstructionState wallState;
        if (!wallStates.TryGetValue(bindingKey, out wallState))
        {
            wallState = new WallConstructionState();
            wallStates[bindingKey] = wallState;
        }

        Material segmentMaterial = SyncWallMaterial(wallState, wallSettings);
        EnsureSegmentCapacity(wallState, orderedPoints.Length - 1);
        for (int i = 0; i < orderedPoints.Length - 1; i++)
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
                displayName,
                i,
                orderedPoints[i],
                orderedPoints[i + 1],
                wallSettings,
                segmentMaterial);
        }

        activeWallBindings.Add(bindingKey);
    }

    public void EndWallRefresh()
    {
        var staleGroups = new List<string>();
        foreach (var entry in wallStates)
        {
            if (!activeWallBindings.Contains(entry.Key))
            {
                DestroySegments(entry.Value);
                staleGroups.Add(entry.Key);
            }
        }

        for (int i = 0; i < staleGroups.Count; i++)
            wallStates.Remove(staleGroups[i]);
    }

    public void ClearWalls()
    {
        foreach (var entry in wallStates)
            DestroySegments(entry.Value);

        wallStates.Clear();
        activeWallBindings.Clear();
    }

    public void ClearAll()
    {
        ClearWalls();

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
        MarkerWallConstructionSettings wallSettings,
        Material material)
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

        bool invertY = wallSettings != null && wallSettings.invertY;
        Vector3 segmentPosition = invertY
            ? new Vector3(wallCenter.x, wallCenter.y - wallHeight, wallCenter.z)
            : wallCenter;

        segment.name = string.Format("Construction_{0}_Wall_{1}", displayName, segmentIndex);
        segment.transform.SetParent(markerParent, false);
        segment.transform.position = segmentPosition;
        segment.transform.rotation = Quaternion.LookRotation(wallLengthDirection, Vector3.up);
        SetWorldScale(segment.transform, markerParent, new Vector3(wallThickness, wallHeight, wallLength));

        MeshRenderer meshRenderer = segment.GetComponent<MeshRenderer>();
        if (meshRenderer != null)
            meshRenderer.sharedMaterial = material;
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
        AddFace(vertexList, triangleList, uvList, corners[2], corners[3], corners[7], corners[6], Vector3.forward);
        AddFace(vertexList, triangleList, uvList, corners[0], corners[3], corners[7], corners[4], Vector3.left);
        AddFace(vertexList, triangleList, uvList, corners[2], corners[1], corners[5], corners[6], Vector3.right);

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

    Material SyncWallMaterial(WallConstructionState wallState, MarkerWallConstructionSettings wallSettings)
    {
        Texture2D texture = wallSettings != null ? wallSettings.texture : null;

        if (texture == null)
        {
            if (wallState.OverrideMaterial != null)
            {
                Object.Destroy(wallState.OverrideMaterial);
                wallState.OverrideMaterial = null;
            }
            return GetOrCreateWallMaterial();
        }

        if (wallState.OverrideMaterial == null || wallState.OverrideMaterial.mainTexture != texture)
        {
            if (wallState.OverrideMaterial != null)
                Object.Destroy(wallState.OverrideMaterial);

            wallState.OverrideMaterial = new Material(GetOrCreateWallMaterial());
            wallState.OverrideMaterial.name = "MarkerConstructionWallMaterial_Textured";
            wallState.OverrideMaterial.mainTexture = texture;
        }

        return wallState.OverrideMaterial;
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

        if (wallState.OverrideMaterial != null)
        {
            Object.Destroy(wallState.OverrideMaterial);
            wallState.OverrideMaterial = null;
        }
    }

    sealed class WallConstructionState
    {
        public readonly List<GameObject> Segments = new List<GameObject>();
        public Material OverrideMaterial;
    }
}
