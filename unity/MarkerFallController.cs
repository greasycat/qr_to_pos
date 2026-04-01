using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public sealed class MarkerFallController : MonoBehaviour
{
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
        Vector3 position = body.position;
        position.x = targetPosition.x;
        position.z = targetPosition.z;

        if (spawnFromAbove)
            position.y = targetPosition.y + fallSpawnHeight;
        else if (position.y < targetPosition.y)
            position.y = targetPosition.y;

        body.position = position;
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
}
