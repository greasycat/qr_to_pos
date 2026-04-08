using System;
using System.Collections.Generic;
using UnityEngine;

public enum MarkerConstructionChoice
{
    Prefab,
    Wall,
}

[Serializable]
public sealed class MarkerWallConstructionSettings
{
    public float height = 1f;
    public float thickness = 0.2f;
}

[Serializable]
public sealed class MarkerConstructionBinding
{
    public string name = "Construction";
    public MarkerConstructionChoice choice = MarkerConstructionChoice.Prefab;
    public List<int> markerIndexes = new List<int>();
    public GameObject prefab;
    public MarkerWallConstructionSettings wall = new MarkerWallConstructionSettings();
}

public sealed class MarkerConstructionAssignment
{
    public readonly string BindingKey;
    public readonly MarkerConstructionBinding Binding;
    public readonly int MarkerIndex;
    public readonly int Order;

    public MarkerConstructionAssignment(string bindingKey, MarkerConstructionBinding binding, int markerIndex, int order)
    {
        BindingKey = bindingKey;
        Binding = binding;
        MarkerIndex = markerIndex;
        Order = order;
    }

    public string DisplayName
    {
        get
        {
            if (Binding != null && !string.IsNullOrEmpty(Binding.name))
                return Binding.name;

            return BindingKey;
        }
    }
}
