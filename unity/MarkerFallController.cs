using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public sealed class MarkerFallController : MonoBehaviour
{
    const float ColliderCastPadding = 0.005f;
    const float MinimumCastDistance = 0.05f;

    float fallSpawnHeight;
    float maxFallSpeed;
    Rigidbody cachedRigidbody;

    public void Configure(float fallSpawnHeight, float fallAcceleration, float maxFallSpeed)
    {
        this.fallSpawnHeight = Mathf.Max(0f, fallSpawnHeight);
        this.maxFallSpeed = Mathf.Max(0.01f, maxFallSpeed);

        Rigidbody body = GetOrCreateRigidbody();
        body.useGravity = true;
        body.isKinematic = false;
        body.interpolation = RigidbodyInterpolation.Interpolate;
        body.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
        body.constraints = RigidbodyConstraints.FreezePositionX
            | RigidbodyConstraints.FreezePositionZ
            | RigidbodyConstraints.FreezeRotation;
    }

    public void SetTarget(Vector3 targetPosition, bool spawnFromAbove)
    {
        Rigidbody body = GetOrCreateRigidbody();
        Vector3 position = targetPosition;

        if (spawnFromAbove)
            position.y = targetPosition.y + fallSpawnHeight;

        body.position = ResolveSettledPosition(position);
        body.linearVelocity = Vector3.zero;
        transform.rotation = Quaternion.identity;
    }

    void FixedUpdate()
    {
        Rigidbody body = GetOrCreateRigidbody();
        Vector3 velocity = body.linearVelocity;
        velocity.x = 0f;
        velocity.z = 0f;
        velocity.y = Mathf.Max(velocity.y, -maxFallSpeed);
        body.linearVelocity = velocity;
    }

    Rigidbody GetOrCreateRigidbody()
    {
        if (cachedRigidbody != null)
            return cachedRigidbody;

        cachedRigidbody = GetComponent<Rigidbody>();
        if (cachedRigidbody == null)
            cachedRigidbody = gameObject.AddComponent<Rigidbody>();

        return cachedRigidbody;
    }

    Vector3 ResolveSettledPosition(Vector3 startPosition)
    {
        Collider markerCollider = GetComponent<Collider>();
        if (markerCollider == null)
            return startPosition;

        Vector3 halfExtents = markerCollider.bounds.extents;
        halfExtents.x = Mathf.Max(ColliderCastPadding, halfExtents.x - ColliderCastPadding);
        halfExtents.y = Mathf.Max(ColliderCastPadding, halfExtents.y - ColliderCastPadding);
        halfExtents.z = Mathf.Max(ColliderCastPadding, halfExtents.z - ColliderCastPadding);

        float castLift = halfExtents.y + ColliderCastPadding;
        Vector3 castCenter = new Vector3(startPosition.x, startPosition.y + castLift, startPosition.z);
        float castDistance = castLift + Mathf.Max(fallSpawnHeight, halfExtents.y + MinimumCastDistance);

        RaycastHit[] hits = Physics.BoxCastAll(
            castCenter,
            halfExtents,
            Vector3.down,
            Quaternion.identity,
            castDistance,
            ~0,
            QueryTriggerInteraction.Ignore);

        bool hasHit = false;
        float nearestDistance = float.PositiveInfinity;
        for (int i = 0; i < hits.Length; i++)
        {
            if (hits[i].collider == null || hits[i].collider == markerCollider)
                continue;

            if (hits[i].distance < nearestDistance)
            {
                nearestDistance = hits[i].distance;
                hasHit = true;
            }
        }

        if (!hasHit)
            return startPosition;

        return castCenter + Vector3.down * nearestDistance;
    }
}
